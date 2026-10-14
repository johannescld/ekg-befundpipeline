"""
PTB-XL 1D-ResNet Training: HYBRIDES Schema (19 Klassen)
- Kopie von train_resnet_ptbxl.py, NUR Label-Ebene + Class-Weighting geändert.
- Labels via label_mapping_hybrid.py (5 Superklassen -> 19 Zielklassen,
  seltene Subklassen als *_other gebündelt).
- Eigene Output-Dateien -> überschreibt NICHT das 5-Klassen-Modell.
- Folds 1-8 Train, F9 Val, F10 Test. AUROC pro Klasse + RF-Baseline.

Unterschiede zum 5-Klassen-Skript (alles markiert mit [HYBRID]):
  1. Labels: build_label_matrix statt aggregate_superdiagnostic
  2. N_CLASSES = 19 (automatisch aus mlb.classes_)
  3. BCEWithLogitsLoss mit pos_weight (Klassen-Ungleichgewicht)
  4. Output: *_hybrid.pt / auroc_comparison_hybrid.csv
"""

import os
import ast
import pickle
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import pywt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# [HYBRID] Label-Schema aus dem separaten Modul
from label_mapping_hybrid import build_label_matrix, TARGET_TO_SUPER, TARGET_LABELS_DE

from ekg_config import BASE_STR, get_device  # zentrale Pfadkonfiguration


# ─── Pfade ────────────────────────────────────────────────────────────────────
DATAFOLDER = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTFOLDER  = BASE_STR + "ptbxl_results/"
os.makedirs(OUTFOLDER, exist_ok=True)

SAMPLING_RATE = 500
TASK          = "hybrid_19"          # [HYBRID] 19 Zielklassen
POS_WEIGHT_CAP = 10.0                # [HYBRID] Obergrenze für Class-Weighting

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
    X = np.array([wfdb.rdsamp(DATAFOLDER + f)[0] for f in tqdm(Y.filename_hr)])
    np.save(raw_cache_npy, X)
    print("Cache gespeichert.")

print(f"Signal-Shape: {X.shape}   (N × Zeitpunkte × Ableitungen)")

# ─── 2. [HYBRID] 19-Klassen-Labels ────────────────────────────────────────────
print("\n=== Berechne Hybrid-Labels (19 Zielklassen) ===")
mlb, y_full = build_label_matrix(Y)            # (N, 19), Reihenfolge = mlb.classes_
CLASS_NAMES = list(mlb.classes_)

# Nur EKGs mit mind. einer diagnostischen Zielklasse (Form/Rhythmus-only verwerfen)
mask    = y_full.sum(axis=1) > 0
X_filt  = X[mask]
Y_filt  = Y[mask].copy()
y_all   = y_full[mask]

print(f"Klassen ({len(CLASS_NAMES)}): {CLASS_NAMES}")
print(f"Verworfen (keine Zielklasse): {int((~mask).sum())}  |  verbleibend: {len(X_filt)}")
print(f"Klassenhäufigkeit (gesamt):   {dict(zip(CLASS_NAMES, y_all.sum(axis=0)))}")

# ─── 3. Fold-Split ────────────────────────────────────────────────────────────
strat_fold = Y_filt.strat_fold.values
X_train = X_filt[strat_fold <= 8];  y_train = y_all[strat_fold <= 8]
X_val   = X_filt[strat_fold == 9];  y_val   = y_all[strat_fold == 9]
X_test  = X_filt[strat_fold == 10]; y_test  = y_all[strat_fold == 10]
print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ─── 4. Signal-Normalisierung (per-Lead Z-Score auf Train-Stats) ──────────────
mean_tr = X_train.mean(axis=(0, 1), keepdims=True)
std_tr  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
X_train_n = (X_train - mean_tr) / std_tr
X_val_n   = (X_val   - mean_tr) / std_tr
X_test_n  = (X_test  - mean_tr) / std_tr

def to_tensor(X, y):
    return torch.FloatTensor(X.transpose(0, 2, 1)), torch.FloatTensor(y)

Xt_tr, yt_tr = to_tensor(X_train_n, y_train)
Xt_va, yt_va = to_tensor(X_val_n,   y_val)
Xt_te, yt_te = to_tensor(X_test_n,  y_test)

BATCH = 32
train_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(Xt_va, yt_va), batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(Xt_te, yt_te), batch_size=BATCH, shuffle=False, num_workers=0)

# ─── 5. 1D-ResNet (identisch zum 5-Klassen-Skript) ────────────────────────────
class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(1)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)

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
            nn.Conv1d(input_channels, inplanes, kernel_size=ks, stride=2, padding=(ks-1)//2, bias=False),
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
        layers = [block(self.inplanes, planes, stride, kernel_size, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(x)

def resnet1d_wang(num_classes, input_channels=12):
    return ResNet1d(BasicBlock1d, [1, 1, 1, 1],
                    input_channels=input_channels, num_classes=num_classes,
                    inplanes=128, kernel_size=5)

# ─── 6. Training ──────────────────────────────────────────────────────────────
device = get_device()
print(f"\n=== Training auf: {device} ===")

N_CLASSES = len(CLASS_NAMES)                         # [HYBRID] = 19
model = resnet1d_wang(num_classes=N_CLASSES).to(device)
print(f"Modell-Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# [HYBRID] Class-Weighting: pos_weight = neg/pos pro Klasse, gedeckelt
pos = y_train.sum(axis=0)
neg = len(y_train) - pos
pos_weight = np.clip(neg / np.maximum(pos, 1), 1.0, POS_WEIGHT_CAP)
print("pos_weight (gedeckelt @ {:.0f}): {}".format(
    POS_WEIGHT_CAP, dict(zip(CLASS_NAMES, np.round(pos_weight, 1)))))
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(pos_weight).to(device))

EPOCHS = 30
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader))

best_val_auc = 0.0
best_model_path = OUTFOLDER + "best_resnet1d_hybrid.pt"   # [HYBRID] eigener Pfad

def predict_loader(loader, mdl, dev):
    mdl.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(torch.sigmoid(mdl(xb.to(dev))).cpu().numpy())
            trues.append(yb.numpy())
    return np.vstack(preds), np.vstack(trues)

def macro_auc(true, pred):
    """Makro-AUROC nur über Klassen mit >=1 Positiv im jeweiligen Split."""
    aucs = []
    for i in range(true.shape[1]):
        if true[:, i].sum() > 0:
            try:
                aucs.append(roc_auc_score(true[:, i], pred[:, i]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")

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

    val_preds, val_true = predict_loader(val_loader, model, device)
    val_auc = macro_auc(val_true, val_preds)

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
    resnet_aurocs[cls] = roc_auc_score(test_true[:, i], test_preds[:, i]) if test_true[:, i].sum() > 0 else float("nan")
resnet_macro = macro_auc(test_true, test_preds)

# ─── 8. Random Forest Baseline (Wavelet-Features) ─────────────────────────────
print("\n=== Random Forest Baseline (Wavelet-Features) ===")

def get_features(vals):
    vals = np.asarray(vals, dtype=float)
    cnt  = np.bincount(np.clip(vals.astype(int), 0, 255)) + 1e-9
    ent  = -((cnt / cnt.sum()) * np.log(cnt / cnt.sum() + 1e-9)).sum()
    zcr  = float(np.diff(np.sign(vals)).astype(bool).sum())
    mcr  = float(np.diff(np.sign(vals - vals.mean())).astype(bool).sum())
    pcts = np.nanpercentile(vals, [5, 25, 50, 75, 95]).tolist()
    return [ent, zcr, mcr] + pcts + [float(np.nanmean(vals)), float(np.nanstd(vals)),
                                      float(np.nanvar(vals)), float(np.nanmean(np.sqrt(vals**2)))]

def ecg_wavelet_features(signal, wavelet="db6"):
    feats = []
    for lead in signal.T:
        for c in pywt.wavedec(lead, wavelet=wavelet, level=3):
            feats.extend(get_features(c))
    return feats

# [HYBRID] rf_features.pkl ist label-unabhängig (nur Signal). Cache nur nutzen,
# wenn die EKG-Auswahl identisch ist (sonst Shape-Mismatch -> neu berechnen).
rf_cache = OUTFOLDER + "rf_features.pkl"
reuse = False
if os.path.exists(rf_cache):
    with open(rf_cache, "rb") as f:
        Ftr, Fva, Fte = pickle.load(f)
    reuse = (Ftr.shape[0] == len(X_train) and Fte.shape[0] == len(X_test))
    print("Gecachte Wavelet-Features", "wiederverwendet." if reuse else "passen nicht (neu berechnen).")
if not reuse:
    print("Berechne Wavelet-Features …")
    Ftr = np.array([ecg_wavelet_features(s) for s in tqdm(X_train, desc="Train")])
    Fva = np.array([ecg_wavelet_features(s) for s in tqdm(X_val,   desc="Val")])
    Fte = np.array([ecg_wavelet_features(s) for s in tqdm(X_test,  desc="Test")])
    with open(rf_cache, "wb") as f:
        pickle.dump((Ftr, Fva, Fte), f)

print(f"Feature-Dimensionalität: {Ftr.shape[1]}")
scaler = StandardScaler()
Ftr_s = scaler.fit_transform(Ftr)
Fte_s = scaler.transform(Fte)

print("Trainiere Random Forest …")
rf = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42), n_jobs=-1)
rf.fit(Ftr_s, y_train)
rf_preds = rf.predict_proba(Fte_s)
if isinstance(rf_preds, list):
    rf_preds = np.column_stack([p[:, 1] for p in rf_preds])

rf_aurocs = {}
for i, cls in enumerate(CLASS_NAMES):
    rf_aurocs[cls] = roc_auc_score(y_test[:, i], rf_preds[:, i]) if y_test[:, i].sum() > 0 else float("nan")
rf_macro = macro_auc(y_test, rf_preds)

# ─── 9. Ergebnis-Tabelle ──────────────────────────────────────────────────────
n_test_pos = y_test.sum(axis=0)
print("\n" + "=" * 78)
print("  AUROC pro Klasse – Test-Set (Fold 10)  [500 Hz, Hybrid-19]")
print("=" * 78)
print(f"{'Klasse':<11} {'Super':<6} {'n_test':>6} {'1D-ResNet':>11} {'RandomForest':>13}")
print("-" * 78)
order = sorted(CLASS_NAMES, key=lambda c: (TARGET_TO_SUPER[c], -n_test_pos[CLASS_NAMES.index(c)]))
for cls in order:
    i = CLASS_NAMES.index(cls)
    print(f"{cls:<11} {TARGET_TO_SUPER[cls]:<6} {int(n_test_pos[i]):>6} "
          f"{resnet_aurocs[cls]:>11.4f} {rf_aurocs[cls]:>13.4f}")
print("-" * 78)
print(f"{'Macro':<11} {'':<6} {int(n_test_pos.sum()):>6} {resnet_macro:>11.4f} {rf_macro:>13.4f}")
print("=" * 78)

results_df = pd.DataFrame({
    "Klasse":         CLASS_NAMES + ["Macro"],
    "Superklasse":    [TARGET_TO_SUPER[c] for c in CLASS_NAMES] + [""],
    "Bezeichnung":    [TARGET_LABELS_DE[c] for c in CLASS_NAMES] + [""],
    "n_test_pos":     [int(n_test_pos[i]) for i in range(len(CLASS_NAMES))] + [int(n_test_pos.sum())],
    "ResNet1d_AUROC": [resnet_aurocs[c] for c in CLASS_NAMES] + [resnet_macro],
    "RF_AUROC":       [rf_aurocs[c]     for c in CLASS_NAMES] + [rf_macro],
})
out_csv = OUTFOLDER + "auroc_comparison_hybrid.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nErgebnisse gespeichert: {out_csv}")
print(f"Modell gespeichert:     {best_model_path}")
