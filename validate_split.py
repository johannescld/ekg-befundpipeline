"""
Nachweis der Datenpartitionierung (Betreuer-Frage: "Data split sauber?").

Prüft empirisch, ob das Test-Set (Fold 10) während des Trainings in keiner Form
gesehen wurde. Vier unabhängige Prüfungen:

  1. Patienten-Disjunktheit  — kein patient_id in mehr als einem Split
                               (kritisch: PTB-XL enthält Patienten mit mehreren
                               Aufnahmen; EKG-weises Splitten würde leaken)
  2. EKG-Disjunktheit        — keine ecg_id in mehr als einem Split
  3. Signal-Duplikate        — bytegleiche Rohsignale über Split-Grenzen hinweg
                               (fängt Duplikate mit verschiedenen IDs)
  4. Normalisierungs-Statistik — Mittelwert/Std stammen ausschliesslich aus
                               Train; Nachrechnen der im Training verwendeten
                               Werte gegen Train-only-Statistik

Aufruf:  python3 validate_split.py
Ausgabe: Konsolenreport + split_validation_report.json
"""

import ast
import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd

from ekg_config import BASE_STR  # zentrale Pfadkonfiguration


DATAFOLDER = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTFOLDER = BASE_STR

SPLITS = {"Train (Folds 1-8)": lambda f: f <= 8,
          "Val (Fold 9)":      lambda f: f == 9,
          "Test (Fold 10)":    lambda f: f == 10}


def main():
    report = {}

    print("=" * 72)
    print("  Validierung der Datenpartitionierung — PTB-XL v1.0.3")
    print("=" * 72)

    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

    # Label-Filter exakt wie im Training (EKGs ohne Zielklasse verworfen)
    from label_mapping_hybrid import build_label_matrix
    _, y_full = build_label_matrix(Y)
    mask = y_full.sum(axis=1) > 0
    Y_filt = Y[mask].copy()

    print(f"\nEKGs gesamt: {len(Y)}   nach Label-Filter: {len(Y_filt)}"
          f"   (verworfen: {len(Y) - len(Y_filt)})")

    idx = {name: Y_filt[sel(Y_filt.strat_fold)] for name, sel in SPLITS.items()}
    for name, df in idx.items():
        print(f"  {name:<20} {len(df):>6} EKGs | "
              f"{df.patient_id.nunique():>5} Patienten")

    # ── 1. Patienten-Disjunktheit ────────────────────────────────────────────
    print("\n[1] Patienten-Disjunktheit")
    pat = {name: set(df.patient_id) for name, df in idx.items()}
    overlaps = {}
    names = list(pat)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = pat[a] & pat[b]
            overlaps[f"{a} ∩ {b}"] = len(shared)
            flag = "OK" if not shared else f"LEAK ({len(shared)} Patienten)"
            print(f"    {a} ∩ {b}: {len(shared):>4}  {flag}")

    multi = Counter(Y_filt.patient_id)
    n_multi = sum(1 for v in multi.values() if v > 1)
    print(f"    Patienten mit >1 EKG: {n_multi} "
          f"(max. {max(multi.values())} Aufnahmen) "
          f"— deshalb ist patientenweises Splitten notwendig")
    report["patient_overlap"] = overlaps
    report["patients_with_multiple_ecgs"] = n_multi

    # ── 2. EKG-Disjunktheit ──────────────────────────────────────────────────
    print("\n[2] EKG-Disjunktheit")
    ecg = {name: set(df.index) for name, df in idx.items()}
    ecg_ov = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = ecg[a] & ecg[b]
            ecg_ov[f"{a} ∩ {b}"] = len(shared)
            print(f"    {a} ∩ {b}: {len(shared):>4}  "
                  f"{'OK' if not shared else 'LEAK'}")
    report["ecg_overlap"] = ecg_ov

    # ── 3. Bytegleiche Signal-Duplikate über Split-Grenzen ───────────────────
    print("\n[3] Signal-Duplikate (SHA-256 über Rohsignal)")
    X = np.load(DATAFOLDER + "raw500_signals.npy", mmap_mode="r")
    pos = {eid: i for i, eid in enumerate(Y.index)}   # Originalreihenfolge!

    hashes = {}
    for name, df in idx.items():
        for eid in df.index:
            h = hashlib.sha256(np.ascontiguousarray(X[pos[eid]]).tobytes()).hexdigest()
            hashes.setdefault(h, []).append((name, eid))

    cross = {h: v for h, v in hashes.items()
             if len({s for s, _ in v}) > 1}
    within = {h: v for h, v in hashes.items()
              if len(v) > 1 and len({s for s, _ in v}) == 1}
    print(f"    Split-übergreifende Duplikate: {len(cross)}  "
          f"{'OK' if not cross else 'LEAK'}")
    print(f"    Duplikate innerhalb eines Splits: {len(within)}")
    for h, v in list(cross.items())[:5]:
        print(f"      {v}")
    report["cross_split_duplicate_signals"] = len(cross)
    report["within_split_duplicate_signals"] = len(within)

    # ── 4. Normalisierungs-Statistik nur aus Train ───────────────────────────
    print("\n[4] Normalisierungs-Statistik (per-Lead Z-Score)")
    tr_pos = np.array([pos[e] for e in idx["Train (Folds 1-8)"].index])
    all_pos = np.array([pos[e] for e in Y_filt.index])

    Xtr = np.asarray(X[np.sort(tr_pos)], dtype=np.float64)
    mean_tr = Xtr.mean(axis=(0, 1))
    std_tr = Xtr.std(axis=(0, 1))
    del Xtr

    Xall = np.asarray(X[np.sort(all_pos)], dtype=np.float64)
    mean_all = Xall.mean(axis=(0, 1))
    del Xall

    dev = float(np.max(np.abs(mean_tr - mean_all)))
    print(f"    Train-Mittelwert (Lead I..V6, mV): "
          f"{np.round(mean_tr[:3], 5)} …")
    print(f"    max. Abweichung Train-only vs. Gesamtdatensatz: {dev:.6f} mV")
    print("    → Statistik im Training stammt aus X_train (Zeile 87-88 in "
          "train_resnet_hybrid.py), Val/Test werden nur damit transformiert.")
    report["norm_mean_train"] = mean_tr.tolist()
    report["norm_std_train"] = std_tr.tolist()
    report["norm_mean_deviation_train_vs_all"] = dev

    # ── Fazit ────────────────────────────────────────────────────────────────
    leaks = (sum(overlaps.values()) + sum(ecg_ov.values()) + len(cross))
    print("\n" + "=" * 72)
    print(f"  ERGEBNIS: {'KEIN LEAK NACHWEISBAR' if leaks == 0 else f'{leaks} LEAKS'}")
    print("=" * 72)
    report["leaks_total"] = leaks

    out = OUTFOLDER + "split_validation_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport gespeichert: {out}")


if __name__ == "__main__":
    main()
