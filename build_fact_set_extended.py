"""
Stufe 2 Vorbereitung (erweitert): Faktum-Sets für je einen korrekt
klassifizierten Repräsentanten ALLER 5 Superklassen + ein Multi-Label-EKG.

Importiert Architektur/Logik 1:1 aus build_fact_set.py (nichts dort verändert),
erweitert nur die Auswahl. Schreibt example_fact_sets.json (Superset, enthält
weiterhin die bisherigen EKGs 9/526/1056 deterministisch).

Aufruf: python3 build_fact_set_extended.py
"""

import ast
import json
import warnings
import numpy as np
import pandas as pd
import torch
from ekg_config import get_device

from build_fact_set import (
    resnet1d_wang, build_fact_set,
    CLASS_NAMES, DATAFOLDER, MODEL_PATH, SR, THRESHOLD, OUTFILE,
)
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("=== Lade PTB-XL Metadaten ===")
    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

    scp_df  = pd.read_csv(DATAFOLDER + "scp_statements.csv", index_col=0)
    diag_df = scp_df[scp_df.diagnostic == 1.0]

    def get_superclasses(scp_dict):
        return list({diag_df.loc[c, "diagnostic_class"]
                     for c in scp_dict
                     if c in diag_df.index and pd.notna(diag_df.loc[c, "diagnostic_class"])})

    Y["superdiagnostic"] = Y.scp_codes.apply(get_superclasses)

    # WICHTIG: Positions-Mapping VOR dem Filtern (X_all = ungefilterte
    # Original-Reihenfolge). Behebt Index-Versatz-Bug (2026-06-30).
    idx_map = {ecg_id: pos for pos, ecg_id in enumerate(Y.index)}

    Y = Y[Y.superdiagnostic.map(len) > 0]

    print("=== Lade gecachte 500-Hz-Signale ===")
    X_all = np.load(DATAFOLDER + "raw500_signals.npy", allow_pickle=False)

    print("=== Berechne Normalisierungsstats (Folds 1-8) ===")
    train_mask = Y.strat_fold <= 8
    train_positions = [idx_map[i] for i in Y.index[train_mask]]
    X_train = X_all[train_positions]
    mean_tr = X_train.mean(axis=(0, 1), keepdims=True)
    std_tr  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    print("=== Lade ResNet-Modell ===")
    device = get_device()
    model  = resnet1d_wang(num_classes=5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"  Modell geladen auf: {device}")

    test_df = Y[Y.strat_fold == 10].copy()
    print(f"\n=== Inferenz über {len(test_df)} Test-EKGs (Fold 10) ===")
    all_probs = {}
    for ecg_id in test_df.index:
        pos         = idx_map[ecg_id]
        signal_norm = (X_all[pos] - mean_tr[0]) / std_tr[0]
        x_t = torch.FloatTensor(signal_norm.T[np.newaxis])
        with torch.no_grad():
            logits = model(x_t.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()[0]
        all_probs[ecg_id] = probs

    def is_correct(ecg_id):
        probs       = all_probs[ecg_id]
        true_labels = set(test_df.loc[ecg_id, "superdiagnostic"])
        pred_labels = {cls for cls, p in zip(CLASS_NAMES, probs) if p >= THRESHOLD}
        if not pred_labels:
            pred_labels = {CLASS_NAMES[int(np.argmax(probs))]}
        return pred_labels == true_labels

    # ─── Auswahl: je 1 korrekt klassifizierter, reiner Repräsentant pro Klasse ──
    selected_ids = []
    for cls in CLASS_NAMES:                       # CD, HYP, MI, NORM, STTC
        reps = [i for i in test_df.index
                if set(test_df.loc[i, "superdiagnostic"]) == {cls} and is_correct(i)]
        if reps:
            selected_ids.append(int(reps[0]))
            print(f"  {cls:5s}: ecg_id {reps[0]}")
        else:
            print(f"  {cls:5s}: kein korrekt klassifizierter reiner Vertreter gefunden")

    # ein Multi-Label-EKG ergänzen
    multi_ok = [i for i in test_df.index
                if len(test_df.loc[i, "superdiagnostic"]) >= 2 and is_correct(i)]
    if multi_ok:
        selected_ids.append(int(multi_ok[0]))
        print(f"  MULTI: ecg_id {multi_ok[0]} "
              f"({sorted(test_df.loc[multi_ok[0], 'superdiagnostic'])})")

    print(f"\n=== {len(selected_ids)} EKG-IDs ausgewählt: {selected_ids} ===")

    # ─── Faktum-Set Erstellung ──────────────────────────────────────────────────
    fact_sets = []
    for ecg_id in selected_ids:
        signal      = X_all[idx_map[ecg_id]]
        probs       = all_probs[ecg_id]
        meta_row    = Y.loc[ecg_id]
        true_labels = sorted(meta_row["superdiagnostic"])
        fs = build_fact_set(ecg_id, meta_row, signal, probs, true_labels)
        fact_sets.append(fs)
        print(f"\n--- EKG {ecg_id} | true={true_labels} "
              f"| pred={fs['predicted_classes']} | conf={fs['confidence']} ---")
        print("  features:", fs["classical_features"])

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(fact_sets, f, indent=2, ensure_ascii=False)
    print(f"\n=== {len(fact_sets)} Faktum-Sets gespeichert: {OUTFILE} ===")
