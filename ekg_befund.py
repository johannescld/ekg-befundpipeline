"""
END-TO-END-TOOL:  EKG-Datei rein  →  Befund raus.

ZWEI Eingabewege:
  A) Beliebige EKG-DATEI reinwerfen (der Normalfall):
       python3 ekg_befund.py --file mein_ekg.hea            # WFDB (.hea/.dat)
       python3 ekg_befund.py --file mein_ekg.csv            # CSV, 12 Spalten
       python3 ekg_befund.py --file mein_ekg.npy            # NumPy (N,12) o. (12,N)
     Optional: --fs 500 (Abtastrate der Datei), --age 65 --sex w
  B) PTB-XL-Beispiel per ID (zum Testen):
       python3 ekg_befund.py --id 271

  Modellwahl:  --hybrid  -> 19-Klassen-Modell (sonst 5 Superklassen)

Kette: Signal laden/anpassen (12 Kanäle, 500 Hz, 5000 Samples)
  → ResNet-Klassifikation → neurokit2-Messwerte → Plausibilitätsprüfung
  → Konfidenz-Gating → Befundtext (API falls Key, sonst Session).
"""

import sys
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings("ignore")

from build_fact_set import (
    resnet1d_wang, extract_classical_features, confidence_label, DATAFOLDER, MODEL_PATH, SR,
)
from plausibility import annotate_fact_set
from label_mapping_hybrid import TARGET_TO_SUPER, TARGET_LABELS_DE, TARGET_CLASSES
from ekg_config import get_device

HYBRID_MODEL_PATH = os.path.join(os.path.dirname(MODEL_PATH), "best_resnet1d_hybrid.pt")
NORM_STATS_PATH   = os.path.join(os.path.dirname(MODEL_PATH), "norm_stats.npz")
THRESHOLD = 0.5
CONF_GATE = 0.65
N_SAMPLES = 5000          # 10 s @ 500 Hz — Modell-Eingabelänge

SUPER_LABELS_DE = {
    "NORM": "Normalbefund", "MI": "Myokardinfarkt (akut oder abgelaufen)",
    "STTC": "ST-Strecken-/T-Wellen-Veränderungen",
    "CD": "Erregungsleitungsstörung", "HYP": "Hypertrophie/Belastung",
}


# ─── Signal-Import: beliebige Datei -> (5000, 12) @ 500 Hz ────────────────────
def load_ecg_file(path, fs_in=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".hea", ".dat", ""):                       # WFDB
        import wfdb
        rec = wfdb.rdsamp(os.path.splitext(path)[0])
        sig, fields = rec[0], rec[1]
        fs = fs_in or fields.get("fs", 500)
    elif ext == ".npy":
        sig = np.load(path, allow_pickle=False); fs = fs_in or 500
    elif ext in (".csv", ".txt"):
        sig = np.genfromtxt(path, delimiter="," if ext == ".csv" else None)
        fs = fs_in or 500
    else:
        raise ValueError(f"Unbekanntes Format: {ext} (unterstützt: WFDB, .npy, .csv, .txt)")

    sig = np.asarray(sig, dtype=float)
    if sig.ndim != 2:
        raise ValueError(f"Signal muss 2D sein, ist {sig.shape}")
    if sig.shape[0] < sig.shape[1]:      # (12, N) -> (N, 12)
        sig = sig.T
    if sig.shape[1] != 12:
        raise ValueError(f"Erwarte 12 Ableitungen, gefunden {sig.shape[1]}. "
                         f"Reihenfolge: I,II,III,aVR,aVL,aVF,V1–V6.")

    if fs and int(fs) != SR:             # auf 500 Hz resampeln
        from scipy.signal import resample
        sig = resample(sig, int(round(sig.shape[0] * SR / fs)), axis=0)

    n = sig.shape[0]                     # auf 5000 Samples bringen (zentriert)
    if n > N_SAMPLES:
        start = (n - N_SAMPLES) // 2
        sig = sig[start:start + N_SAMPLES]
    elif n < N_SAMPLES:
        pad = N_SAMPLES - n
        sig = np.pad(sig, ((pad // 2, pad - pad // 2), (0, 0)), mode="edge")
    return sig, fs


def get_norm_stats():
    """Per-Lead Z-Score-Stats aus Trainingsfolds (1-8). Gecacht in norm_stats.npz."""
    if os.path.exists(NORM_STATS_PATH):
        d = np.load(NORM_STATS_PATH)
        return d["mean"], d["std"]
    import ast
    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    X = np.load(DATAFOLDER + "raw500_signals.npy", allow_pickle=False)
    train_pos = [i for i, e in enumerate(Y.index) if Y.loc[e, "strat_fold"] <= 8]
    mean = X[train_pos].mean(axis=(0, 1), keepdims=True)
    std  = X[train_pos].std(axis=(0, 1), keepdims=True) + 1e-8
    np.savez(NORM_STATS_PATH, mean=mean, std=std)
    return mean, std


def befund_diagnosen(class_names, probs, hybrid):
    preds = [(c, float(p)) for c, p in zip(class_names, probs) if p >= THRESHOLD]
    if not preds:
        c = class_names[int(np.argmax(probs))]; preds = [(c, float(np.max(probs)))]
    diagnosen, gating = [], []
    for cls, p in sorted(preds, key=lambda x: -x[1]):
        if not hybrid:
            diagnosen.append(SUPER_LABELS_DE.get(cls, cls)); continue
        if p >= CONF_GATE:
            diagnosen.append(TARGET_LABELS_DE[cls])
        else:
            fb = SUPER_LABELS_DE.get(TARGET_TO_SUPER[cls], TARGET_TO_SUPER[cls])
            diagnosen.append(fb)
            gating.append(f"{cls} nur p={p:.2f} < {CONF_GATE} → Fallback '{fb}'")
    seen, uniq = set(), []
    for d in diagnosen:
        if d not in seen:
            seen.add(d); uniq.append(d)
    return uniq, gating


def load_signal_from_id(ecg_id):
    import ast
    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    if ecg_id not in Y.index:
        sys.exit(f"FEHLER: ecg_id {ecg_id} nicht in PTB-XL.")
    X = np.load(DATAFOLDER + "raw500_signals.npy", allow_pickle=False)
    pos = {e: i for i, e in enumerate(Y.index)}[ecg_id]      # volle Reihenfolge!
    meta = Y.loc[ecg_id]
    patient = {"age_years": int(meta["age"]) if not pd.isna(meta["age"]) else None,
               # PTB-XL: 0 = männlich, 1 = weiblich (vgl. build_fact_set.py)
               "sex": "weiblich" if meta["sex"] == 1 else "männlich"}
    return X[pos], patient


def main():
    ap = argparse.ArgumentParser(description="EKG-Datei rein → Befund raus")
    ap.add_argument("--file", help="Pfad zur EKG-Datei (WFDB/.csv/.npy)")
    ap.add_argument("--id", type=int, help="PTB-XL ecg_id (Testmodus)")
    ap.add_argument("--hybrid", action="store_true", help="19-Klassen-Modell")
    ap.add_argument("--fs", type=float, help="Abtastrate der Datei (Hz), Standard 500")
    ap.add_argument("--age", type=int, help="Patientenalter (nur --file)")
    ap.add_argument("--sex", choices=["m", "w"], help="Geschlecht (nur --file)")
    args = ap.parse_args()

    if bool(args.file) == bool(args.id):
        sys.exit("Bitte GENAU eine Quelle angeben: --file <pfad>  ODER  --id <ecg_id>.")

    hybrid = args.hybrid
    model_path = HYBRID_MODEL_PATH if hybrid else MODEL_PATH
    class_names = TARGET_CLASSES if hybrid else ["CD", "HYP", "MI", "NORM", "STTC"]
    print(f"=== EKG-Befund | Schema: {'19 Klassen (Hybrid)' if hybrid else '5 Superklassen'} ===")
    if hybrid and not os.path.exists(model_path):
        sys.exit(f"FEHLER: Hybrid-Modell fehlt ({model_path}). Erst train_resnet_hybrid.py laufen lassen.")

    # ─── Signal beschaffen ───────────────────────────────────────────────────
    source, true_classes = "", []
    if args.file:
        if not os.path.exists(args.file):
            sys.exit(f"FEHLER: Datei nicht gefunden: {args.file}")
        sig, fs = load_ecg_file(args.file, args.fs)
        patient = {"age_years": args.age,
                   "sex": {"m": "männlich", "w": "weiblich"}.get(args.sex)}
        source = f"Datei {os.path.basename(args.file)} (fs={fs} Hz)"
    else:
        sig, patient = load_signal_from_id(args.id)
        source = f"PTB-XL ecg_id {args.id}"

    # ─── Modell + Normalisierung + Inferenz ──────────────────────────────────
    mean_tr, std_tr = get_norm_stats()
    device = get_device()
    model = resnet1d_wang(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    sig_n = (sig - mean_tr[0]) / std_tr[0]
    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.FloatTensor(sig_n.T[np.newaxis]).to(device))).cpu().numpy()[0]

    prob_dict = {c: round(float(p), 3) for c, p in zip(class_names, probs)}
    pred_classes = [c for c, p in zip(class_names, probs) if p >= THRESHOLD] \
        or [class_names[int(np.argmax(probs))]]

    # ─── Messwerte + Plausibilität + Gating ──────────────────────────────────
    feats = extract_classical_features(sig, sr=SR)
    fs_out = {
        "quelle": source,
        "schema": "hybrid_19" if hybrid else "superdiagnostic_5",
        "patient": patient,
        "acquisition": {"sampling_rate_hz": SR, "duration_s": round(sig.shape[0]/SR, 1),
                        "n_leads": int(sig.shape[1])},
        "classical_features": feats,
        "resnet_probabilities": prob_dict,
        "predicted_classes": pred_classes,
        "confidence": confidence_label(float(np.max(probs))),
        "pipeline_stage": "stufe_1_output",
    }
    fs_out = annotate_fact_set(fs_out)
    diagnosen, gating = befund_diagnosen(class_names, probs, hybrid)
    fs_out["befund_diagnosen"] = diagnosen
    if gating:
        fs_out["konfidenz_gating"] = gating

    # ─── Ausgabe ─────────────────────────────────────────────────────────────
    print(f"Quelle: {source}")
    print(f"Patient: {patient['age_years']} J, {patient['sex']}")
    print(f"Stufe-1-Vorhersage: {pred_classes}")
    print("Top-Wahrscheinlichkeiten:",
          {k: v for k, v in sorted(prob_dict.items(), key=lambda x: -x[1])[:5]})
    print(f"Messwerte: { {k:(round(v,1) if isinstance(v,(int,float)) else v) for k,v in feats.items()} }")
    pl = fs_out["plausibility"]
    print("Plausibilität:", "✓ alle plausibel" if pl["alle_verwertbar"]
          else f"⚠ {pl['n_artefakte']} Artefakt(e) verworfen")
    for a in pl["artefakte"]:
        print("   →", a)
    for g in gating:
        print("   Gating →", g)
    print(f"\nBefund-Diagnose(n): {diagnosen}")

    out = os.path.join(os.path.dirname(MODEL_PATH), "befund_input_last.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fs_out, f, indent=2, ensure_ascii=False)

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n→ API-Key gefunden — generiere Befund …")
        from llm_befund import build_user_prompt, SYSTEM_PROMPT, MODEL
        import anthropic, re
        resp = anthropic.Anthropic().messages.create(
            model=MODEL, max_tokens=1024, temperature=0.2,
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": build_user_prompt(fs_out)}])
        raw = resp.content[0].text
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        print("\n=== BEFUND ===")
        print(json.dumps(json.loads(m.group(0) if m else raw), indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 64)
        print(f"  Faktum-Set gespeichert: {out}")
        print("  → Kein API-Key: Befundtext wird in der Claude-Session geschrieben.")
        print("=" * 64)


if __name__ == "__main__":
    main()
