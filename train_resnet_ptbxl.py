"""
PTB-XL 1D-ResNet Training: Superdiagnostic (5 Klassen)
- 500 Hz Signale (records500)
- Folds 1-8 Training, F9 Validation, F10 Test
- AUROC pro Klasse + Vergleich mit Random Forest Baseline
"""

import os
import sys
import ast
import pickle
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import pywt
import scipy.stats
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ekg_config import BASE_STR, get_device  # zentrale Pfadkonfiguration


# ─── Pfade ────────────────────────────────────────────────────────────────────
DATAFOLDER = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTFOLDER  = BASE_STR + "ptbxl_results/"
os.makedirs(OUTFOLDER, exist_ok=True)

SAMPLING_RATE = 500
TASK          = "superdiagnostic"   # → 5 Klassen: NORM, MI, STTC, CD, HYP

# ─── 1. Daten laden ───────────────────────────────────────────────────────────
print("\n=== Lade PTB-XL Metadaten ===")
Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

raw_cache_npy = DATAFOLDER + "raw500_signals.npy"
if os.path.exists(raw_cache_npy):
    print("Lade gecachte 500-Hz-Signale …")
    X = np.load(raw_cache_npy, allow_pickle=False)
else:
    print("Lese 500-Hz-WFDB-Signale (dauert ~5 min) …")
    X = np.array([wfdb.rdsamp(DATAFOLDER + f)[0]
                  for f in tqdm(Y.filename_hr)])
    np.save(raw_cache_npy, X)
    print("Cache gespeichert.")

print(f"Signal-Shape: {X.shape}   (N × Zeitpunkte × Ableitungen)")

# ─── 2. Superdiagnostische Labels ─────────────────────────────────────────────
print("\n=== Berechne Superdiagnostik-Labels ===")
scp_df = pd.read_csv(DATAFOLDER + "scp_statements.csv", index_col=0)
diag_df = scp_df[scp_df.diagnostic == 1.0]

def aggregate_superdiagnostic(scp_dict):
    classes = []
    for code in scp_dict:
        if code in diag_df.index:
            cls = diag_df.loc[code, "diagnostic_class"]
            if pd.notna(cls):
                classes.append(cls)
    return list(set(classes))

Y["superdiagnostic"] = Y.scp_codes.apply(aggregate_superdiagnostic)
Y["superdiagnostic_len"] = Y.superdiagnostic.apply(len)

# Nur Samples mit mindestens einer Klasse
mask = Y.superdiagnostic_len > 0
X_filt = X[mask]
Y_filt = Y[mask].copy()

# One-hot Encoding
mlb = MultiLabelBinarizer()
mlb.fit(Y_filt.superdiagnostic)
y_all = mlb.transform(Y_filt.superdiagnostic)
CLASS_NAMES = list(mlb.classes_)
print(f"Klassen ({len(CLASS_NAMES)}): {CLASS_NAMES}")
print(f"Klassenhäufigkeit (gesamt): {y_all.sum(axis=0)}")

# ─── 3. Fold-Split ────────────────────────────────────────────────────────────
strat_fold = Y_filt.strat_fold.values
X_train = X_filt[strat_fold <= 8]
y_train = y_all[strat_fold <= 8]
X_val   = X_filt[strat_fold == 9]
y_val   = y_all[strat_fold == 9]
X_test  = X_filt[strat_fold == 10]
y_test  = y_all[strat_fold == 10]

print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ─── 4. Signal-Normalisierung ─────────────────────────────────────────────────
# Per-Lead Z-Score auf Trainings-Statistiken
mean_tr = X_train.mean(axis=(0, 1), keepdims=True)
std_tr  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

X_train_n = (X_train - mean_tr) / std_tr
X_val_n   = (X_val   - mean_tr) / std_tr
X_test_n  = (X_test  - mean_tr) / std_tr

# PyTorch: (N, Leads, Zeitpunkte)
def to_tensor(X, y):
    Xt = torch.FloatTensor(X.transpose(0, 2, 1))   # N, 12, 5000
    yt = torch.FloatTensor(y)
    return Xt, yt

Xt_tr, yt_tr = to_tensor(X_train_n, y_train)
Xt_va, yt_va = to_tensor(X_val_n,   y_val)
Xt_te, yt_te = to_tensor(X_test_n,  y_test)

BATCH = 32
train_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(Xt_va, yt_va), batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(Xt_te, yt_te), batch_size=BATCH, shuffle=False, num_workers=0)

# ─── 5. 1D-ResNet (ohne fastai) ───────────────────────────────────────────────
class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(1)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)

def create_head(nf, nc, ps=0.5):
    """Classification head mit Concat-Pooling."""
    return nn.Sequential(
        AdaptiveConcatPool1d(),
        nn.Flatten(),
        nn.BatchNorm1d(2 * nf),
        nn.Dropout(ps / 2),
        nn.Linear(2 * nf, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(ps),
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
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        return self.relu(out + residual)

class ResNet1d(nn.Module):
    def __init__(self, block, layers, input_channels=12, num_classes=5,
                 inplanes=64, kernel_size=3, stride=2):
        super().__init__()
        self.inplanes = inplanes
        ks = kernel_size

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, inplanes, kernel_size=ks,
                      stride=2, padding=(ks-1)//2, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, inplanes,        layers[0], stride=1,      kernel_size=ks)
        self.layer2 = self._make_layer(block, inplanes,        layers[1], stride=stride, kernel_size=ks)
        self.layer3 = self._make_layer(block, inplanes,        layers[2], stride=stride, kernel_size=ks)
        self.layer4 = self._make_layer(block, inplanes,        layers[3], stride=stride, kernel_size=ks)
        self.head   = create_head(self.inplanes, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, kernel_size, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

def resnet1d_wang(num_classes, input_channels=12):
    """ResNet1d Wang (kernel_size=5/3, wie im Paper)."""
    # Wang-ResNet: fix_feature_dim=True → alle Layer gleiche Breite
    return ResNet1d(BasicBlock1d, [1, 1, 1, 1],
                    input_channels=input_channels,
                    num_classes=num_classes,
                    inplanes=128,
                    kernel_size=5)

# ─── 6. Training ──────────────────────────────────────────────────────────────
device = get_device()
print(f"\n=== Training auf: {device} ===")

N_CLASSES = len(CLASS_NAMES)
model = resnet1d_wang(num_classes=N_CLASSES).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Modell-Parameter: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=30, steps_per_epoch=len(train_loader))
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 30
best_val_auc = 0.0
best_model_path = OUTFOLDER + "best_resnet1d.pt"

def predict_loader(loader, mdl, dev):
    mdl.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = mdl(xb.to(dev))
            preds.append(torch.sigmoid(logits).cpu().numpy())
            trues.append(yb.numpy())
    return np.vstack(preds), np.vstack(trues)

print(f"\nEpoche  | Train Loss | Val AUROC (macro)")
print("-" * 45)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(xb)

    avg_loss = total_loss / len(Xt_tr)

    # Validation AUROC
    val_preds, val_true = predict_loader(val_loader, model, device)
    try:
        val_auc = roc_auc_score(val_true, val_preds, average="macro")
    except Exception:
        val_auc = float("nan")

    marker = ""
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), best_model_path)
        marker = " ← best"

    print(f"  {epoch:3d}/{EPOCHS} | {avg_loss:.4f}     | {val_auc:.4f}{marker}")

# ─── 7. Test-Evaluation ResNet ────────────────────────────────────────────────
print(f"\n=== Test-Evaluation (bestes Modell: Val-AUC {best_val_auc:.4f}) ===")
model.load_state_dict(torch.load(best_model_path, weights_only=True))
test_preds, test_true = predict_loader(test_loader, model, device)

resnet_aurocs = {}
for i, cls in enumerate(CLASS_NAMES):
    try:
        auc = roc_auc_score(test_true[:, i], test_preds[:, i])
    except Exception:
        auc = float("nan")
    resnet_aurocs[cls] = auc

resnet_macro = roc_auc_score(test_true, test_preds, average="macro")

# ─── 8. Random Forest Baseline ────────────────────────────────────────────────
print("\n=== Random Forest Baseline (Wavelet-Features) ===")

def get_features(vals):
    vals = np.asarray(vals, dtype=float)
    cnt  = np.bincount(np.clip(vals.astype(int), 0, 255)) + 1e-9
    ent  = -((cnt / cnt.sum()) * np.log(cnt / cnt.sum() + 1e-9)).sum()
    zcr  = float(np.diff(np.sign(vals)).astype(bool).sum())
    mcr  = float(np.diff(np.sign(vals - vals.mean())).astype(bool).sum())
    pcts = np.nanpercentile(vals, [5, 25, 50, 75, 95]).tolist()
    return [ent, zcr, mcr] + pcts + [float(np.nanmean(vals)),
                                      float(np.nanstd(vals)),
                                      float(np.nanvar(vals)),
                                      float(np.nanmean(np.sqrt(vals**2)))]

def ecg_wavelet_features(signal, wavelet="db6"):
    feats = []
    for lead in signal.T:  # (T, 12) → iterate over leads
        coeffs = pywt.wavedec(lead, wavelet=wavelet, level=3)  # level=3 (schneller)
        for c in coeffs:
            feats.extend(get_features(c))
    return feats

rf_cache = OUTFOLDER + "rf_features.pkl"
if os.path.exists(rf_cache):
    print("Lade gecachte Wavelet-Features …")
    with open(rf_cache, "rb") as f:
        Ftr, Fva, Fte = pickle.load(f)
else:
    print("Berechne Wavelet-Features (Train) …")
    Ftr = np.array([ecg_wavelet_features(sig) for sig in tqdm(X_train)])
    print("Berechne Wavelet-Features (Val) …")
    Fva = np.array([ecg_wavelet_features(sig) for sig in tqdm(X_val)])
    print("Berechne Wavelet-Features (Test) …")
    Fte = np.array([ecg_wavelet_features(sig) for sig in tqdm(X_test)])
    with open(rf_cache, "wb") as f:
        pickle.dump((Ftr, Fva, Fte), f)
    print("Features gespeichert.")

print(f"Feature-Dimensionalität: {Ftr.shape[1]}")

# Skalieren
scaler = StandardScaler()
Ftr_s = scaler.fit_transform(Ftr)
Fte_s = scaler.transform(Fte)

print("Trainiere Random Forest …")
rf = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=100, max_depth=None,
                           n_jobs=-1, random_state=42),
    n_jobs=-1)
rf.fit(Ftr_s, y_train)

rf_preds = rf.predict_proba(Fte_s)
# OVR gibt Liste zurück → stapeln
if isinstance(rf_preds, list):
    rf_preds = np.column_stack([p[:, 1] for p in rf_preds])

rf_aurocs = {}
for i, cls in enumerate(CLASS_NAMES):
    try:
        auc = roc_auc_score(y_test[:, i], rf_preds[:, i])
    except Exception:
        auc = float("nan")
    rf_aurocs[cls] = auc

rf_macro = roc_auc_score(y_test, rf_preds, average="macro")

# ─── 9. Ergebnis-Tabelle ──────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  AUROC pro Klasse – Test-Set (Fold 10)  [500 Hz]")
print("=" * 62)
print(f"{'Klasse':<10}  {'1D-ResNet':>12}  {'Random Forest':>14}")
print("-" * 42)
for cls in CLASS_NAMES:
    print(f"{cls:<10}  {resnet_aurocs[cls]:>12.4f}  {rf_aurocs[cls]:>14.4f}")
print("-" * 42)
print(f"{'Macro':.<10}  {resnet_macro:>12.4f}  {rf_macro:>14.4f}")
print("=" * 62)

# Ergebnisse als CSV speichern
results_df = pd.DataFrame({
    "Klasse":        CLASS_NAMES + ["Macro"],
    "ResNet1d_AUROC": [resnet_aurocs[c] for c in CLASS_NAMES] + [resnet_macro],
    "RF_AUROC":       [rf_aurocs[c]     for c in CLASS_NAMES] + [rf_macro],
})
results_df.to_csv(OUTFOLDER + "auroc_comparison.csv", index=False)
print(f"\nErgebnisse gespeichert: {OUTFOLDER}auroc_comparison.csv")
