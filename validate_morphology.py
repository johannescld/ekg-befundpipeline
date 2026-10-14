"""
Validierung der morphologischen Messgrößen gegen die PTB-XL-Diagnosen.

Die in ecg_morphology.py berechneten Größen gehen ungeprüft in den
Befundtext ein — ein falscher Messwert erzeugt einen falschen Befund.
Dieses Skript prüft daher, ob jede Messgröße den lehrbuchmässig
erwarteten Zusammenhang zur unabhängig vergebenen Kardiologen-Diagnose
zeigt.

Geprüfte Hypothesen (jeweils auf dem gesamten Test-Set, Fold 10):

  H1  LAFB/LPFB (Hemiblock)      -> überdrehter Linkstyp (Achse < -30°)
  H2  IMI (Hinterwandinfarkt)    -> pathologische Q-Zacken inferior
  H3  AMI (Vorderwandinfarkt)    -> pathologische Q-Zacken anteroseptal
  H4  CLBBB / CRBBB              -> QRS-Verbreiterung >= 120 ms
  H5  STTC / ISC_ (Repolarisation) -> ST-Senkung oder T-Negativierung
  H6  NORM                       -> unauffällig in allen Kategorien

Ausgabe je Hypothese: Rate im Label-positiven vs. Label-negativen Kollektiv,
Sensitivität, Spezifität und das Verhältnis der Raten. Ein Merkmal, das im
positiven Kollektiv nicht deutlich häufiger auftritt, misst nicht das,
was es messen soll.

Aufruf:  python3 validate_morphology.py [--limit N]
Ausgabe: ptbxl_results/morphology_validation.json
"""

import argparse
import ast
import json
import sys

import numpy as np
import pandas as pd

from ecg_morphology import analyze_morphology
from label_mapping_hybrid import build_label_matrix

from ekg_config import BASE_STR  # zentrale Pfadkonfiguration


DATAFOLDER = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTFOLDER = BASE_STR + "ptbxl_results/"


def merkmale(res: dict) -> dict:
    """Reduziert das Analyseergebnis auf binäre Prüfmerkmale."""
    if not res.get("beurteilbar"):
        return {}
    lag = res.get("lagetyp", {})
    q = res.get("q_zacken", {})
    rp = res.get("r_progression", {})
    st = res.get("st_strecke", {})
    t = res.get("t_wellen", {})
    iv = res.get("intervalle_aus_fiducials", {})
    axis = lag.get("achse_grad")
    qrs = iv.get("qrs_dauer_ms")

    return {
        # Rohwerte für die Nachvollziehbarkeit
        "achse_grad": axis,
        "qrs_dauer_ms": qrs,
        # Ein nicht messbarer Wert ist FEHLEND, nicht 'unauffällig' — sonst
        # zählt jede fehlgeschlagene Messung als Negativbefund und verzerrt
        # Sensitivität und Spezifität.
        "ueberdrehter_linkstyp": (None if axis is None else bool(axis < -30)),
        "q_inferior": "inferior" in (q.get("betroffene_gruppen") or []),
        "q_anteroseptal": "anteroseptal" in (q.get("betroffene_gruppen") or []),
        "q_lateral": "lateral" in (q.get("betroffene_gruppen") or []),
        "qrs_verbreitert": (None if qrs is None else bool(qrs >= 120)),
        "st_senkung": bool(st.get("relevante_senkung")),
        "st_hebung": bool(st.get("relevante_hebung")),
        "t_negativ": bool(t.get("relevante_negativierung")),
        "repolarisationsstoerung": bool(st.get("relevante_senkung")
                                        or t.get("relevante_negativierung")),
        "rs_umschlag_normal": bool(rp.get("rs_umschlag_normal")),
        "unauffaellig": not any([
            q.get("pardee_q_vorhanden"), st.get("relevante_hebung"),
            st.get("relevante_senkung"), t.get("relevante_negativierung"),
            bool(qrs is not None and qrs >= 120),
        ]),
    }


def quote_messbar(df, spalte):
    """Anteil der EKGs, für die eine Messgröße überhaupt bestimmbar war."""
    return float(df[spalte].notna().mean())


# vergleich="norm": Negativkollektiv sind ausschliesslich EKGs mit NORM-Label.
# Notwendig für alle Repolarisationshypothesen — im Gesamtkollektiv stecken
# ISC_, ISCA, ISCI und NST_, die selbst ST-T-Veränderungen tragen. Ein
# Vergleich gegen "alles ausser STTC" misst dann Kranke gegen Kranke und
# erzeugt scheinbar fehlenden Zusammenhang.
HYPOTHESEN = [
    ("H1", "LAFB/LPFB", "ueberdrehter_linkstyp", "Hemiblock → überdrehter Linkstyp", "alle"),
    ("H2", "IMI", "q_inferior", "Hinterwandinfarkt → Q inferior", "norm"),
    ("H3", "AMI", "q_anteroseptal", "Vorderwandinfarkt → Q anteroseptal", "norm"),
    ("H4", "CLBBB", "qrs_verbreitert", "Linksschenkelblock → QRS ≥ 120 ms", "norm"),
    ("H4b", "CRBBB", "qrs_verbreitert", "Rechtsschenkelblock → QRS ≥ 120 ms", "norm"),
    ("H5", "STTC", "repolarisationsstoerung", "ST-T-Veränderung → ST-Senkung/T-negativ", "norm"),
    ("H5b", "ISC_", "repolarisationsstoerung", "Ischämie → ST-Senkung/T-negativ", "norm"),
    ("H6", "NORM", "unauffaellig", "Normalbefund → unauffällige Morphologie", "alle"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="nur die ersten N Test-EKGs (Schnelltest)")
    args = ap.parse_args()

    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    mlb, y_full = build_label_matrix(Y)
    classes = list(mlb.classes_)

    mask = y_full.sum(axis=1) > 0
    Y_f = Y[mask].copy()
    y_f = y_full[mask]
    pos_all = np.array([i for i, k in enumerate(mask) if k])

    is_test = Y_f.strat_fold.values == 10
    test_ids = Y_f.index.values[is_test]
    test_pos = pos_all[is_test]
    y_test = y_f[is_test]
    if args.limit:
        test_ids, test_pos, y_test = (test_ids[:args.limit],
                                      test_pos[:args.limit],
                                      y_test[:args.limit])

    X = np.load(DATAFOLDER + "raw500_signals.npy", mmap_mode="r")

    print(f"=== Morphologische Analyse von {len(test_ids)} Test-EKGs ===")
    rows, n_fail = [], 0
    for k, (eid, p) in enumerate(zip(test_ids, test_pos), 1):
        row = Y_f.loc[eid]
        sex = "weiblich" if row.sex == 1 else "männlich"
        res = analyze_morphology(np.asarray(X[p], dtype=float), 500, sex, row.age)
        m = merkmale(res)
        if not m:
            n_fail += 1
        rows.append({"ecg_id": int(eid), **m})
        if k % 250 == 0:
            print(f"  {k}/{len(test_ids)} …")

    df = pd.DataFrame(rows).set_index("ecg_id")
    print(f"\nNicht auswertbar: {n_fail} von {len(test_ids)} "
          f"({n_fail/len(test_ids):.1%})")

    ylab = pd.DataFrame(y_test, index=test_ids, columns=classes)
    df = df.reindex(ylab.index)

    print("\n" + "=" * 100)
    print("  Validierung: Merkmalsrate im Label-positiven vs. -negativen Kollektiv")
    print("=" * 100)
    print(f"{'':4} {'Diagnose':<11} {'Merkmal':<26} {'n+':>5} "
          f"{'messb.':>7} {'Rate+':>7} {'Rate-':>7} {'Verh.':>7} "
          f"{'Sens':>6} {'Spez':>6}  {'Vergl.':<6}")
    print("-" * 100)

    report = []
    for hid, label, merkmal, beschreibung, vergleich in HYPOTHESEN:
        if label not in ylab.columns or merkmal not in df.columns:
            continue
        pos = ylab[label] == 1
        neg = (ylab["NORM"] == 1) & ~pos if vergleich == "norm" else ~pos
        valid = df[merkmal].notna()
        p, n = pos & valid, neg & valid
        if p.sum() == 0 or n.sum() == 0:
            continue
        rate_p = float(df.loc[p, merkmal].mean())
        rate_n = float(df.loc[n, merkmal].mean())
        ratio = rate_p / rate_n if rate_n > 0 else float("inf")
        sens = rate_p
        spez = 1 - rate_n
        messbar = quote_messbar(df, merkmal)
        print(f"{hid:<4} {label:<11} {merkmal:<26} {int(p.sum()):>5} "
              f"{messbar:>7.1%} {rate_p:>7.1%} {rate_n:>7.1%} {ratio:>7.1f} "
              f"{sens:>6.2f} {spez:>6.2f}  {vergleich}")
        report.append({"hypothese": hid, "beschreibung": beschreibung,
                       "label": label, "merkmal": merkmal,
                       "vergleichskollektiv": vergleich,
                       "n_positiv": int(p.sum()), "n_negativ": int(n.sum()),
                       "anteil_messbar": messbar,
                       "rate_positiv": rate_p,
                       "rate_negativ": rate_n, "verhaeltnis": ratio,
                       "sensitivitaet": sens, "spezifitaet": spez})

    print("-" * 100)
    print("Lesart: 'Verh.' = wie oft häufiger das Merkmal bei vorliegender Diagnose")
    print("auftritt. Werte nahe 1 bedeuten, dass das Merkmal nicht misst, was es soll.")
    print("Niedrige Sensitivität ist erwartbar (nicht jeder Infarkt zeigt ein Q),")
    print("ein niedriges Verhältnis dagegen ist ein Messfehler.")

    out = OUTFOLDER + "morphology_validation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_ekg": len(test_ids), "n_nicht_auswertbar": n_fail,
                   "hypothesen": report}, f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert: {out}")

    df.to_csv(OUTFOLDER + "morphology_features_test.csv")
    print(f"Merkmalstabelle: {OUTFOLDER}morphology_features_test.csv")


if __name__ == "__main__":
    main()
