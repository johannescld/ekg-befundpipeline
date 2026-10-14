"""
Faktum-Set Version 2 — Eingang für das überarbeitete Befundschema.

Führt zusammen, was der lehrbuchkonforme Befund braucht:

  Stufe 1   19-Klassen-ResNet          -> Wahrscheinlichkeiten je Zielklasse
  klassisch neurokit2 (Ableitung II)   -> Frequenz, PQ, QRS, QT, QTc
  NEU       ecg_morphology.py          -> Lagetyp, Q-Zacken, R-Progression,
                                          ST-Strecke, T-Wellen, Extrasystolen
  Stufe 3   plausibility.py            -> Messartefakte verwerfen
            Konfidenz-Gating           -> Rückfall auf die Superklasse
            befund_schema.py           -> sprachliche Sicherheitsstufe

Der wesentliche Unterschied zu Version 1: Die Wahrscheinlichkeiten werden
hier bereits in Sprache übersetzt ("Wahrscheinlich ein Vorderwandinfarkt").
Das Sprachmodell bekommt keine Prozentzahlen mehr zu sehen und kann sie
daher auch nicht in den Befundtext schreiben.

Aufruf:  python3 build_fact_set_v2.py [--ids 9 175 271] [--out DATEI]
"""

import argparse
import ast
import json
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from build_fact_set import (resnet1d_wang, extract_classical_features,
                            confidence_label)
from ecg_morphology import analyze_morphology
from plausibility import annotate_fact_set
from befund_schema import (formuliere_diagnose, sicherheitsstufe,
                           SUPER_LABELS_DE)
from label_mapping_hybrid import (build_label_matrix, TARGET_LABELS_DE,

                                  TARGET_TO_SUPER)

from ekg_config import BASE_STR, get_device  # zentrale Pfadkonfiguration

DATAFOLDER = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTFOLDER = BASE_STR
MODEL_PATH = OUTFOLDER + "ptbxl_results/best_resnet1d_hybrid.pt"

SR = 500
BEFUND_SCHWELLE = 0.50      # darunter erscheint eine Klasse nicht im Befund
GATING_SCHWELLE = 0.65      # darunter Rückfall auf die Superklasse


def _p_superklasse(probs: dict, sup: str) -> float:
    """
    Wahrscheinlichkeit, dass mindestens eine Zielklasse der Superklasse
    zutrifft (Noisy-OR: 1 - Π(1-p_i)).

    Die naheliegende Summe wäre keine Wahrscheinlichkeit — sie kann 1
    überschreiten und täuscht nach Deckelung Gewissheit vor. Noisy-OR
    bleibt in [0,1] und ist für eine einzelne Klasse identisch mit deren
    Wahrscheinlichkeit. Die dabei unterstellte Unabhängigkeit der Klassen
    ist eine Näherung (die Zielklassen sind korreliert) und in der Arbeit
    als solche zu benennen.
    """
    gegen = 1.0
    for k, v in probs.items():
        if TARGET_TO_SUPER[k] == sup:
            gegen *= (1.0 - v)
    return 1.0 - gegen



def pruefe_qrs_dauer(feats: dict, morph: dict, probs: dict) -> dict:
    """
    Kreuzprüfung der QRS-Dauer aus zwei unabhängigen Messwegen.

    Weg 1: Delineation der Ableitung II (neurokit2, build_fact_set)
    Weg 2: Betrag über alle 12 Ableitungen (ecg_morphology)

    Die Absolutwerte beider Wege unterscheiden sich systematisch (Weg 2
    unterschätzt schmale Komplexe um etwa 20 ms), deshalb wird nicht die
    Differenz geprüft, sondern die klinische Einstufung schmal/verbreitert.

    Zusätzlich wird gegen die Klassifikation geprüft: Ein kompletter
    Schenkelblock ist definitionsgemäß mit einer QRS-Dauer ab 120 ms
    verbunden. Meldet die Klassifikation einen solchen, während beide
    Messwege einen schmalen Komplex ausweisen, ist die Messung fraglich —
    nachgewiesen an EKG 180 (Linksschenkelblock, Weg 1 misst 60 ms).

    Ein nicht verwertbarer Wert wird aus dem Faktum-Set entfernt, damit im
    Befund "nicht bestimmbar" steht statt "normwertig".
    """
    w1 = feats.get("qrs_duration_ms")
    w2 = (morph.get("intervalle_aus_fiducials") or {}).get("qrs_dauer_ms")
    breit1 = None if w1 is None else w1 >= 120
    breit2 = None if w2 is None else w2 >= 120

    grund = None
    if w1 is None and w2 is None:
        grund = "beide Messwege ohne Ergebnis"
    elif w2 is None:
        grund = ("zweiter Messweg (12-Ableitungs-Betrag) ohne Ergebnis — "
                 "typisch bei Schenkelblock, wo die Repolarisation nahtlos "
                 "in den Kammerkomplex übergeht")
    elif breit1 is not None and breit1 != breit2:
        grund = (f"Messwege widersprechen sich: {w1:.0f} ms gegen {w2:.0f} ms "
                 f"(einmal schmal, einmal verbreitert)")
    else:
        # Gegenprobe mit der Klassifikation
        block = max((probs.get(k, 0.0) for k in ("CLBBB", "CRBBB")), default=0.0)
        if block >= 0.85 and not (breit1 or breit2):
            grund = ("Klassifikation weist einen kompletten Schenkelblock aus, "
                     "beide Messwege ergeben jedoch einen schmalen Komplex")

    return {"messweg_ableitung_II_ms": w1,
            "messweg_12_ableitungen_ms": w2,
            "verwertbar": grund is None,
            "grund": grund}


def diagnose_block(probs: dict) -> dict:
    """
    Übersetzt Klassenwahrscheinlichkeiten in Befundsprache.

    Drei Regeln, die ein Multi-Label-Modell allein nicht leistet:

    1. AUSSCHLUSS: 'Normalbefund' und eine pathologische Diagnose schliessen
       einander aus. Liegt irgendeine pathologische Klasse über der
       Befundschwelle, wird NORM verworfen — sonst entstehen Befunde wie
       "Nachweis eines Hinterwandinfarktes. Vereinbar mit einem Normalbefund."

    2. EINE DIAGNOSE JE SUPERKLASSE: Von mehreren Zielklassen derselben
       Superklasse (z. B. ISC_ und ISCA) wird nur die wahrscheinlichste
       genannt. Ein Befund mit sechs Diagnosen ist klinisch unbrauchbar.

    3. KONFIDENZ-GATING: Liegt die gewählte Zielklasse unter GATING_SCHWELLE,
       wird auf die Superklasse zurückgefallen — lieber "Myokardinfarkt" als
       ein falsch lokalisierter "Vorderwandinfarkt". Für NORM entfällt das,
       da NORM zugleich Ziel- und Superklasse ist.
    """
    ueber = {k: p for k, p in probs.items() if p >= BEFUND_SCHWELLE}
    formulierungen, gemeldet, gating, verworfen = [], [], [], []

    # ── Regel 1: Normalbefund vs. Pathologie — der stärkere Befund gewinnt ──
    vorbehalt = None
    pathologisch = {k: p for k, p in ueber.items() if k != "NORM"}
    if pathologisch and "NORM" in ueber:
        p_norm = ueber["NORM"]
        top_path, p_path = max(pathologisch.items(), key=lambda x: x[1])
        if p_norm > p_path:
            # Normalbefund dominiert: schwächere Pathologie verwerfen, aber
            # als Vorbehalt vermerken — sie verschwindet nicht spurlos.
            for k, p in pathologisch.items():
                verworfen.append({"klasse": k, "p": round(p, 3),
                                  "grund": f"schwächer als NORM ({p_norm:.3f})"})
            ueber = {"NORM": p_norm}
            vorbehalt = (f"Nebenbefundlich lag {TARGET_LABELS_DE.get(top_path, top_path)} "
                         f"knapp über der Befundschwelle, jedoch deutlich unterhalb "
                         f"des Normalbefundes.")
        else:
            verworfen.append({"klasse": "NORM", "p": round(p_norm, 3),
                              "grund": f"schwächer als {top_path} ({p_path:.3f}) — "
                                       f"Pathologie schliesst Normalbefund aus"})
            ueber = pathologisch

    # ── Regel 2: je Superklasse nur die stärkste Zielklasse ─────────────────
    beste = {}
    for k, p in ueber.items():
        sup = TARGET_TO_SUPER[k]
        if sup not in beste or p > beste[sup][1]:
            if sup in beste:
                verworfen.append({"klasse": beste[sup][0],
                                  "p": round(beste[sup][1], 3),
                                  "grund": f"schwächer als {k} in derselben "
                                           f"Superklasse {sup}"})
            beste[sup] = (k, p)
        else:
            verworfen.append({"klasse": k, "p": round(p, 3),
                              "grund": f"schwächer als {beste[sup][0]} in "
                                       f"derselben Superklasse {sup}"})

    # ── Regel 3: Konfidenz-Gating, dann Formulierung ────────────────────────
    for sup, (klasse, p) in sorted(beste.items(), key=lambda x: -x[1][1]):
        ziel, p_ziel, ebene = klasse, p, "zielklasse"

        # Gating nur für echte Unterklassen. NORM und STTC sind zugleich
        # Ziel- und Superklasse; ein Rückfall auf sich selbst wertet die
        # Aussage über die Geschwisterklassen auf, ohne dass sich die
        # Formulierung ändert — der Leser sähe die Aufwertung nicht.
        if p < GATING_SCHWELLE and klasse != TARGET_TO_SUPER[klasse]:
            p_sup = _p_superklasse(probs, sup)
            gating.append({"von": klasse, "auf": sup,
                           "p_subklasse": round(p, 3),
                           "p_superklasse": round(p_sup, 3)})
            ziel, p_ziel, ebene = sup, p_sup, "superklasse (Konfidenz-Gating)"

        txt = formuliere_diagnose(ziel, p_ziel)
        if txt:
            formulierungen.append(txt)
            gemeldet.append({"klasse": ziel,
                             "bezeichnung": TARGET_LABELS_DE.get(
                                 ziel, SUPER_LABELS_DE.get(ziel, ziel)),
                             "p": round(p_ziel, 3),
                             "stufe": sicherheitsstufe(p_ziel),
                             "ebene": ebene})

    if not formulierungen:
        top = max(probs.items(), key=lambda x: x[1])
        formulierungen.append("Keine Diagnose erreicht die Befundschwelle; "
                              "eingeschränkt beurteilbares EKG")
        gemeldet.append({"klasse": top[0],
                         "bezeichnung": TARGET_LABELS_DE.get(top[0]),
                         "p": round(top[1], 3), "stufe": None,
                         "ebene": "unterhalb der Befundschwelle"})

    block = {"formulierungen": formulierungen, "gemeldet": gemeldet,
             "konfidenz_gating": gating, "verworfen": verworfen}
    if vorbehalt:
        block["vorbehalt"] = vorbehalt
    return block


def build(ecg_id, meta_row, signal, probs_arr, class_names, true_labels):
    probs = {c: float(p) for c, p in zip(class_names, probs_arr)}
    sex = "weiblich" if meta_row["sex"] == 1 else "männlich"
    age = int(meta_row["age"]) if not pd.isna(meta_row["age"]) else None

    feats = extract_classical_features(signal, sr=SR)
    morph = analyze_morphology(signal, sr=SR, sex=sex, age=age,
                               qrs_ms=feats.get("qrs_duration_ms"))

    qrs_pruefung = pruefe_qrs_dauer(feats, morph, probs)

    fs = {
        "ecg_id": int(ecg_id),
        "schema_version": 2,
        "patient": {"age_years": age, "sex": sex},
        "acquisition": {"sampling_rate_hz": SR,
                        "duration_s": round(signal.shape[0] / SR, 1),
                        "n_leads": int(signal.shape[1])},
        "classical_features": feats,
        "morphologie": morph,
        "diagnose": diagnose_block(probs),
        "true_classes": true_labels,
        # Nur für die Auswertung — nicht Teil des Prompts:
        "qualitaetssicherung": {
            "resnet_probabilities": {k: round(v, 3) for k, v in
                                     sorted(probs.items(), key=lambda x: -x[1])
                                     if v >= 0.10},
            "confidence": confidence_label(float(np.max(probs_arr))),
            "befund_schwelle": BEFUND_SCHWELLE,
            "gating_schwelle": GATING_SCHWELLE,
        },
        "qrs_kreuzpruefung": qrs_pruefung,
        "pipeline_stage": "stufe_1_output_v2",
    }
    fs = annotate_fact_set(fs)
    if not qrs_pruefung["verwertbar"]:
        # Wert aus dem verwertbaren Satz entfernen: Der Befund muss "nicht
        # bestimmbar" sagen, nicht "normwertig".
        fs.setdefault("classical_features_verwertbar", {})["qrs_duration_ms"] = None
    return fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+",
                    default=[9, 65, 175, 128, 271, 7670],
                    help="PTB-XL ecg_ids")
    ap.add_argument("--out", default=OUTFOLDER + "example_fact_sets_v2.json")
    args = ap.parse_args()

    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    mlb, y_full = build_label_matrix(Y)
    class_names = list(mlb.classes_)

    idx_map = {e: i for i, e in enumerate(Y.index)}       # Originalreihenfolge
    X = np.load(DATAFOLDER + "raw500_signals.npy", mmap_mode="r")

    # Normalisierung mit Trainingsstatistik (Folds 1-8, gefiltert wie im Training)
    mask = y_full.sum(axis=1) > 0
    Y_f = Y[mask]
    pos_all = np.array([i for i, k in enumerate(mask) if k])
    tr = Y_f.strat_fold.values <= 8
    Xtr = np.asarray(X[pos_all[tr]], dtype=np.float32)
    mean_tr = Xtr.mean(axis=(0, 1), keepdims=True)
    std_tr = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
    del Xtr

    device = get_device()
    model = resnet1d_wang(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device,
                                     weights_only=True))
    model.eval()

    label_lookup = pd.DataFrame(y_full, index=Y.index, columns=class_names)

    out = []
    for eid in args.ids:
        sig = np.asarray(X[idx_map[eid]], dtype=float)
        norm = (sig - mean_tr[0]) / std_tr[0]
        with torch.no_grad():
            p = torch.sigmoid(model(torch.FloatTensor(
                norm.T[np.newaxis]).to(device))).cpu().numpy()[0]
        true = [c for c in class_names if label_lookup.loc[eid, c] == 1]
        fs = build(eid, Y.loc[eid], sig, p, class_names, true)
        out.append(fs)

        d = fs["diagnose"]
        print(f"\nEKG {eid}  (wahr: {', '.join(true) or '—'})")
        for f in d["formulierungen"]:
            print(f"   → {f}")
        for g in d["konfidenz_gating"]:
            print(f"   [Gating] {g['von']} (p={g['p_subklasse']}) "
                  f"→ {g['auf']} (p={g['p_superklasse']})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert: {args.out}")


if __name__ == "__main__":
    main()
