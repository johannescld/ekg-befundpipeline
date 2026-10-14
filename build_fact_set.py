"""
Stufe 2 Vorbereitung: Strukturiertes Faktum-Set für 3 Beispiel-EKGs
- Lädt best_resnet1d.pt und erzeugt Klassenwahrscheinlichkeiten
- Extrahiert klassische Features via neurokit2 (500 Hz)
- Gibt JSON-Faktum-Sets aus, die direkt an ein LLM übergeben werden können

Aufruf: python build_fact_set.py
"""

import ast
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
import neurokit2 as nk

from ekg_config import BASE_STR, get_device  # zentrale Pfadkonfiguration

warnings.filterwarnings("ignore")

# ─── Pfade ────────────────────────────────────────────────────────────────────
DATAFOLDER  = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
MODEL_PATH  = BASE_STR + "ptbxl_results/best_resnet1d.pt"
OUTFILE     = BASE_STR + "example_fact_sets.json"

SR          = 500
CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]   # alphabetisch (MultiLabelBinarizer)
THRESHOLD   = 0.5

# ─── ResNet-Architektur (identisch zu train_resnet_ptbxl.py) ──────────────────
class AdaptiveConcatPool1d(nn.Module):
    def forward(self, x):
        return torch.cat([nn.AdaptiveMaxPool1d(1)(x),
                          nn.AdaptiveAvgPool1d(1)(x)], dim=1)

def create_head(nf, nc, ps=0.5):
    return nn.Sequential(
        AdaptiveConcatPool1d(), nn.Flatten(),
        nn.BatchNorm1d(2 * nf), nn.Dropout(ps / 2),
        nn.Linear(2 * nf, 512), nn.ReLU(inplace=True),
        nn.BatchNorm1d(512), nn.Dropout(ps),
        nn.Linear(512, nc),
    )

class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        k2 = kernel_size // 2 + 1
        self.conv2 = nn.Conv1d(planes, planes, k2, padding=(k2-1)//2, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.downsample(x) if self.downsample else x
        return self.relu(out + residual)

class ResNet1d(nn.Module):
    def __init__(self, block, layers, input_channels=12, num_classes=5,
                 inplanes=64, kernel_size=3, stride=2):
        super().__init__()
        self.inplanes = inplanes
        ks = kernel_size
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, inplanes, ks, stride=2, padding=(ks-1)//2, bias=False),
            nn.BatchNorm1d(inplanes), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, inplanes, layers[0], stride=1,      kernel_size=ks)
        self.layer2 = self._make_layer(block, inplanes, layers[1], stride=stride, kernel_size=ks)
        self.layer3 = self._make_layer(block, inplanes, layers[2], stride=stride, kernel_size=ks)
        self.layer4 = self._make_layer(block, inplanes, layers[3], stride=stride, kernel_size=ks)
        self.head   = create_head(self.inplanes, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers_list = [block(self.inplanes, planes, stride, kernel_size, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(self.inplanes, planes, kernel_size=kernel_size))
        return nn.Sequential(*layers_list)

    def forward(self, x):
        for layer in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
        return self.head(x)

def resnet1d_wang(num_classes=5, input_channels=12):
    return ResNet1d(BasicBlock1d, [1, 1, 1, 1],
                    input_channels=input_channels, num_classes=num_classes,
                    inplanes=128, kernel_size=5)

# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────
def confidence_label(max_prob: float) -> str:
    if max_prob >= 0.85:
        return "hoch"
    if max_prob >= 0.65:
        return "mittel"
    return "niedrig"

def extract_classical_features(signal: np.ndarray, sr: int = 500) -> dict:
    """Klassische Intervall-Features aus Lead II (Index 1) via neurokit2."""
    lead_ii = signal[:, 1]
    try:
        signals, info = nk.ecg_process(lead_ii, sampling_rate=sr)
        hr  = signals["ECG_Rate"].mean()
        rp  = info.get("ECG_R_Peaks", [])
        rr_intervals = np.diff(rp) / sr * 1000 if len(rp) > 1 else [None]

        waves = nk.ecg_delineate(lead_ii, info, sampling_rate=sr, method="dwt")[1]

        def mean_interval(starts, ends):
            vals = []
            for s, e in zip(starts or [], ends or []):
                if s and e and not (np.isnan(s) or np.isnan(e)):
                    vals.append((e - s) / sr * 1000)
            return round(float(np.mean(vals)), 1) if vals else None

        pr_ms  = mean_interval(waves.get("ECG_P_Onsets"),  waves.get("ECG_R_Onsets"))
        qrs_ms = mean_interval(waves.get("ECG_R_Onsets"),  waves.get("ECG_R_Offsets"))
        qt_ms  = mean_interval(waves.get("ECG_R_Onsets"),  waves.get("ECG_T_Offsets"))

        qtc = None
        if qt_ms and rr_intervals[0]:
            rr_mean_s = np.mean([r for r in rr_intervals if r]) / 1000
            qtc = round(qt_ms / np.sqrt(rr_mean_s), 1)

        return {
            "hr_bpm":         round(float(hr), 1) if not np.isnan(hr) else None,
            "pr_interval_ms": pr_ms,
            "qrs_duration_ms": qrs_ms,
            "qt_interval_ms": qt_ms,
            "qtc_bazett_ms":  qtc,
        }
    except Exception as e:
        return {"error": f"Feature-Extraktion fehlgeschlagen: {e}"}

def build_fact_set(ecg_id: int, meta_row: pd.Series, signal: np.ndarray,
                   probs: np.ndarray, true_labels: list) -> dict:
    """Baut das strukturierte Faktum-Set für ein EKG."""
    prob_dict   = {cls: round(float(p), 3) for cls, p in zip(CLASS_NAMES, probs)}
    pred_labels = [cls for cls, p in zip(CLASS_NAMES, probs) if p >= THRESHOLD]
    if not pred_labels:
        pred_labels = [CLASS_NAMES[int(np.argmax(probs))]]  # Fallback: argmax

    feats = extract_classical_features(signal, sr=SR)

    return {
        "ecg_id": int(ecg_id),
        "patient": {
            "age_years": int(meta_row["age"]) if not pd.isna(meta_row["age"]) else None,
            # PTB-XL kodiert sex: 0 = männlich, 1 = weiblich
            # (belegt über height/weight: sex 0 im Mittel 173.5 cm / 78.5 kg,
            #  sex 1 161.0 cm / 64.6 kg). Vor dem 15.08.2026 war dies vertauscht.
            "sex": "weiblich" if meta_row["sex"] == 1 else "männlich",
        },
        "acquisition": {
            "sampling_rate_hz": SR,
            "duration_s": round(signal.shape[0] / SR, 1),
            "n_leads": signal.shape[1],
        },
        "classical_features": feats,
        "resnet_probabilities": prob_dict,
        "predicted_classes": pred_labels,
        "true_classes": true_labels,
        "multi_label": len(pred_labels) > 1,
        "confidence": confidence_label(float(np.max(probs))),
        "pipeline_stage": "stufe_1_output",
    }

# ─── Hauptprogramm ────────────────────────────────────────────────────────────
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

    # WICHTIG: X_all (raw500_signals.npy) ist in der ORIGINAL-Reihenfolge der
    # vollständigen Datenbank. Das Positions-Mapping MUSS daher VOR dem Filtern
    # gebaut werden — sonst lädt man für jede ecg_id nach einem gedroppten EKG
    # das falsche Signal (Index-Versatz-Bug, behoben 2026-06-30).
    idx_map = {ecg_id: pos for pos, ecg_id in enumerate(Y.index)}

    Y = Y[Y.superdiagnostic.map(len) > 0]

    print("=== Lade gecachte 500-Hz-Signale ===")
    X_all = np.load(DATAFOLDER + "raw500_signals.npy", allow_pickle=False)

    # Normalisierungsstatistiken aus Trainingsfolds (1-8) berechnen
    print("=== Berechne Normalisierungsstats (Folds 1-8) ===")
    train_mask = Y.strat_fold <= 8
    train_positions = [idx_map[i] for i in Y.index[train_mask]]
    X_train = X_all[train_positions]
    mean_tr = X_train.mean(axis=(0, 1), keepdims=True)
    std_tr  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    print(f"  Mean shape: {mean_tr.shape}, Std range: [{std_tr.min():.4f}, {std_tr.max():.4f}]")

    # ─── Modell laden ─────────────────────────────────────────────────────────
    print("=== Lade ResNet-Modell ===")
    device = get_device()
    model  = resnet1d_wang(num_classes=5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"  Modell geladen auf: {device}")

    # ─── Inferenz über alle Test-EKGs (Fold 10) ──────────────────────────────
    test_df = Y[Y.strat_fold == 10].copy()
    print(f"\n=== Inferenz über {len(test_df)} Test-EKGs (Fold 10) ===")

    all_probs  = {}
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

    # ─── Auswahl: korrekt klassifizierte Repräsentanten ──────────────────────
    norm_ok  = [i for i in test_df.index
                if test_df.loc[i, "superdiagnostic"] == ["NORM"] and is_correct(i)]
    mi_ok    = [i for i in test_df.index
                if set(test_df.loc[i, "superdiagnostic"]) == {"MI"} and is_correct(i)]
    multi_ok = [i for i in test_df.index
                if len(test_df.loc[i, "superdiagnostic"]) >= 2 and is_correct(i)]

    selected_ids = [int(norm_ok[0]), int(mi_ok[0]), int(multi_ok[0])]
    print(f"\n=== Korrekt klassifizierte EKG-IDs: {selected_ids} ===")
    for eid in selected_ids:
        print(f"  ecg_id {eid}: true={sorted(test_df.loc[eid, 'superdiagnostic'])}")

    # ─── Faktum-Set Erstellung ────────────────────────────────────────────────
    fact_sets = []
    for ecg_id in selected_ids:
        signal   = X_all[idx_map[ecg_id]]
        probs    = all_probs[ecg_id]
        meta_row = Y.loc[ecg_id]
        true_labels = sorted(meta_row["superdiagnostic"])
        fs = build_fact_set(ecg_id, meta_row, signal, probs, true_labels)
        fact_sets.append(fs)

        print(f"\n--- EKG {ecg_id} ---")
        print(json.dumps(fs, indent=2, ensure_ascii=False))

    # ─── Speichern ────────────────────────────────────────────────────────────
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(fact_sets, f, indent=2, ensure_ascii=False)
    print(f"\n=== Faktum-Sets gespeichert: {OUTFILE} ===")
