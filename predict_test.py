"""
Reproduziert die Test-Vorhersagen des Hybrid-19-Modells und speichert sie,
damit alle weiteren Auswertungen (Bootstrap-KI, Fmax, Average Precision,
Granularitätsanalyse) ohne erneuten Modell-Lauf möglich sind.

Pipeline identisch zu train_resnet_hybrid.py:
  Label-Filter -> Fold-Split -> Z-Score mit TRAIN-Statistik -> ResNet1d

Aufruf:  python3 predict_test.py
Ausgabe: ptbxl_results/test_predictions_hybrid.npz
         (preds, true, ecg_ids, class_names)
"""

import ast
import numpy as np
import pandas as pd
import torch

from label_mapping_hybrid import build_label_matrix
# Architektur aus build_fact_set (identisch zu train_resnet_hybrid, aber
# import-sicher — das Trainingsskript hat keinen __main__-Schutz).
from build_fact_set import resnet1d_wang

from ekg_config import BASE_STR, get_device  # zentrale Pfadkonfiguration


DATAFOLDER = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTFOLDER = BASE_STR + "ptbxl_results/"


def main():
    print("=== Lade Metadaten ===")
    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

    mlb, y_full = build_label_matrix(Y)
    class_names = list(mlb.classes_)
    mask = y_full.sum(axis=1) > 0
    Y_filt = Y[mask].copy()
    y_all = y_full[mask]

    # Position in der ORIGINAL-Reihenfolge der Signaldatei (Index-Bug-Fix)
    pos_all = np.array([i for i, keep in enumerate(mask) if keep])

    fold = Y_filt.strat_fold.values
    tr, te = fold <= 8, fold == 10

    X = np.load(DATAFOLDER + "raw500_signals.npy", mmap_mode="r")

    print("=== Normalisierungs-Statistik aus Train ===")
    Xtr = np.asarray(X[pos_all[tr]], dtype=np.float32)
    mean_tr = Xtr.mean(axis=(0, 1), keepdims=True)
    std_tr = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
    del Xtr

    Xte = (np.asarray(X[pos_all[te]], dtype=np.float32) - mean_tr) / std_tr
    y_te = y_all[te]
    ecg_ids = Y_filt.index.values[te]
    print(f"Test: {Xte.shape[0]} EKGs, {len(class_names)} Klassen")

    device = get_device()
    model = resnet1d_wang(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(OUTFOLDER + "best_resnet1d_hybrid.pt",
                                     weights_only=True, map_location=device))
    model.eval()

    print(f"=== Inferenz auf {device} ===")
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 32):
            xb = torch.FloatTensor(Xte[i:i + 32].transpose(0, 2, 1)).to(device)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    preds = np.vstack(preds)

    out = OUTFOLDER + "test_predictions_hybrid.npz"
    np.savez_compressed(out, preds=preds, true=y_te,
                        ecg_ids=ecg_ids, class_names=np.array(class_names))
    print(f"Gespeichert: {out}")

    # Sanity-Check gegen die dokumentierte Macro-AUROC (0.9272)
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(y_te[:, i], preds[:, i])
            for i in range(len(class_names)) if y_te[:, i].sum() > 0]
    print(f"Macro-AUROC (Kontrolle): {np.mean(aucs):.4f}  "
          f"[dokumentiert: 0.9272]")


if __name__ == "__main__":
    main()
