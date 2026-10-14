"""
Analyse der 20 am häufigsten falsch klassifizierten EKGs (ResNet1d, Fold 10 Test-Set)
- Lädt Modell + Daten identisch zu train_resnet_ptbxl.py
- Berechnet per-Sample Fehlermaß (BCE-Loss + L1-Abweichung)
- Plottet Signale mit Ground-Truth- und Predicted-Labels
"""

import os, ast, sys, warnings
import numpy as np
import pandas as pd
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import Counter

from ekg_config import BASE_STR, get_device  # zentrale Pfadkonfiguration

warnings.filterwarnings("ignore")

# ─── Pfade ─────────────────────────────────────────────────────────────────────
DATAFOLDER = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTFOLDER  = BASE_STR + "ptbxl_results/"
PLOTFOLDER = BASE_STR + "ptbxl_results/error_plots/"
os.makedirs(PLOTFOLDER, exist_ok=True)

# ─── Modell-Definition (identisch zu train_resnet_ptbxl.py) ────────────────────
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
        nn.BatchNorm1d(2*nf), nn.Dropout(ps/2),
        nn.Linear(2*nf, 512), nn.ReLU(inplace=True),
        nn.BatchNorm1d(512), nn.Dropout(ps),
        nn.Linear(512, nc),
    )

class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        pad = (kernel_size-1)//2
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        k2 = kernel_size//2+1
        self.conv2 = nn.Conv1d(planes, planes, k2, padding=(k2-1)//2, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample: residual = self.downsample(x)
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
            nn.MaxPool1d(3, stride=2, padding=1),
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
                nn.Conv1d(self.inplanes, planes*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes*block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, kernel_size, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(x)

def resnet1d_wang(num_classes, input_channels=12):
    return ResNet1d(BasicBlock1d, [1,1,1,1], input_channels=input_channels,
                    num_classes=num_classes, inplanes=128, kernel_size=5)

# ─── 1. Daten laden ────────────────────────────────────────────────────────────
print("=== Lade Metadaten & Signale ===")
Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

raw_cache = DATAFOLDER + "raw500_signals.npy"
X = np.load(raw_cache, allow_pickle=False)
print(f"Signale geladen: {X.shape}")

# ─── 2. Labels ─────────────────────────────────────────────────────────────────
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

mask   = Y.superdiagnostic_len > 0
X_filt = X[mask]
Y_filt = Y[mask].copy()

mlb = MultiLabelBinarizer()
mlb.fit(Y_filt.superdiagnostic)
y_all = mlb.transform(Y_filt.superdiagnostic)
CLASS_NAMES = list(mlb.classes_)
print(f"Klassen: {CLASS_NAMES}")

# ─── 3. Test-Split (Fold 10) ───────────────────────────────────────────────────
strat_fold = Y_filt.strat_fold.values
train_mask = strat_fold <= 8
test_mask  = strat_fold == 10

X_train = X_filt[train_mask]
X_test  = X_filt[test_mask]
y_test  = y_all[test_mask]
Y_test_meta = Y_filt[test_mask].copy()

# Normalisierung
mean_tr = X_train.mean(axis=(0,1), keepdims=True)
std_tr  = X_train.std(axis=(0,1),  keepdims=True) + 1e-8
X_test_n = (X_test - mean_tr) / std_tr

# ─── 4. Modell laden & Inference ───────────────────────────────────────────────
device = get_device()
print(f"\n=== Inference auf: {device} ===")

model = resnet1d_wang(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load(OUTFOLDER + "best_resnet1d.pt", weights_only=True))
model.eval()

Xt = torch.FloatTensor(X_test_n.transpose(0, 2, 1))  # N, 12, 5000
preds_list = []
with torch.no_grad():
    for i in range(0, len(Xt), 64):
        batch = Xt[i:i+64].to(device)
        preds_list.append(torch.sigmoid(model(batch)).cpu().numpy())
y_pred = np.vstack(preds_list)

# ─── 5. Per-Sample Fehlermaß ───────────────────────────────────────────────────
# BCE-Loss pro Sample (zeigt wie "überrascht" das Modell ist)
eps = 1e-7
bce_per_sample = -( y_test * np.log(y_pred + eps)
                  + (1 - y_test) * np.log(1 - y_pred + eps) ).mean(axis=1)

# L1-Abweichung (einfach interpretierbar)
l1_per_sample = np.abs(y_pred - y_test).mean(axis=1)

# Kombiniertes Maß: ranke nach BCE
error_score = bce_per_sample
sorted_idx  = np.argsort(error_score)[::-1]
top20_idx   = sorted_idx[:20]

print(f"\nTop-20 Fehlklassifikationen (höchster BCE-Loss):")
print(f"{'Rang':>4} {'ecg_id':>8} {'BCE':>7} {'GT-Labels':<30} {'Pred-Probs'}")
print("-"*80)
for rank, idx in enumerate(top20_idx, 1):
    ecg_id  = Y_test_meta.index[idx]
    gt_labs = [CLASS_NAMES[j] for j in range(len(CLASS_NAMES)) if y_test[idx,j]==1]
    probs   = {CLASS_NAMES[j]: f"{y_pred[idx,j]:.2f}" for j in range(len(CLASS_NAMES))}
    print(f"{rank:>4} {ecg_id:>8} {error_score[idx]:>7.3f}  {str(gt_labs):<30}  {probs}")

# ─── 6. Analyse: Muster ────────────────────────────────────────────────────────
print("\n=== Musteeranalyse ===")
n_labels_top20 = [y_test[i].sum() for i in top20_idx]
print(f"Multi-Label (≥2 Klassen): {sum(1 for n in n_labels_top20 if n>=2)}/20")
print(f"Klassen-Verteilung GT in Top-20:")
all_gt = [CLASS_NAMES[j] for i in top20_idx for j in range(len(CLASS_NAMES)) if y_test[i,j]==1]
for cls, cnt in Counter(all_gt).most_common():
    print(f"  {cls}: {cnt}x")

# SCP-Codes der Top-20 (zeigt Subklassen / Grenzfälle)
print("\nSCP-Codes der Top-20 EKGs:")
for rank, idx in enumerate(top20_idx, 1):
    ecg_id = Y_test_meta.index[idx]
    row    = Y_test_meta.loc[ecg_id]
    print(f"  #{rank:2d} ecg_id={ecg_id}: {dict(row.scp_codes)}")

# ─── 7. Signal-Plots ───────────────────────────────────────────────────────────
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
FS = 500
T  = np.arange(X_test.shape[1]) / FS  # Zeitachse in Sekunden

print(f"\n=== Plotte 20 EKG-Signale → {PLOTFOLDER} ===")

def plot_ecg(signal, ecg_id, gt_labels, pred_probs, rank, bce, filename):
    """12-Kanal EKG Plot mit Label-Info."""
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0d1117')

    # Farbpalette
    GRID_COLOR   = '#21262d'
    SIGNAL_COLOR = '#58a6ff'
    TITLE_COLOR  = '#e6edf3'
    GT_COLOR     = '#3fb950'   # grün für True
    FP_COLOR     = '#f85149'   # rot für False Positive
    FN_COLOR     = '#d29922'   # orange für False Negative

    gs = gridspec.GridSpec(13, 1, hspace=0.08,
                           top=0.88, bottom=0.04, left=0.06, right=0.99)

    # Header
    ax_hdr = fig.add_subplot(gs[0])
    ax_hdr.set_facecolor('#161b22')
    ax_hdr.axis('off')

    # Ground-Truth und Prediction als farbige Labels
    gt_set   = set(gt_labels)
    pred_set = set(c for c,p in pred_probs.items() if p >= 0.5)

    label_text = ""
    for cls in CLASS_NAMES:
        p = pred_probs[cls]
        in_gt   = cls in gt_set
        in_pred = cls in pred_set
        if in_gt and in_pred:     col = GT_COLOR   # TP
        elif in_gt and not in_pred: col = FN_COLOR  # FN
        elif not in_gt and in_pred: col = FP_COLOR  # FP
        else:                       col = '#484f58'  # TN
        label_text += f"  {cls}:p={p:.2f}"

    ax_hdr.text(0.01, 0.75,
        f"Rang #{rank}  |  ecg_id={ecg_id}  |  BCE-Loss={bce:.3f}",
        color=TITLE_COLOR, fontsize=13, fontweight='bold',
        transform=ax_hdr.transAxes, va='top')

    # Legende
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=GT_COLOR, label='TP (GT✓ Pred✓)'),
        Patch(facecolor=FN_COLOR, label='FN (GT✓ Pred✗)'),
        Patch(facecolor=FP_COLOR, label='FP (GT✗ Pred✓)'),
        Patch(facecolor='#484f58', label='TN (GT✗ Pred✗)'),
    ]
    ax_hdr.legend(handles=legend_items, loc='upper right',
                  framealpha=0.3, fontsize=9, ncol=4)

    # Label-Balken
    bar_x = 0.01
    for cls in CLASS_NAMES:
        p = pred_probs[cls]
        in_gt   = cls in gt_set
        in_pred = cls in pred_set
        if in_gt and in_pred:       col = GT_COLOR
        elif in_gt and not in_pred: col = FN_COLOR
        elif not in_gt and in_pred: col = FP_COLOR
        else:                       col = '#484f58'
        ax_hdr.text(bar_x, 0.20, f"{cls}\n{p:.2f}",
                    color=col, fontsize=9, fontweight='bold',
                    transform=ax_hdr.transAxes, va='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#161b22', ec=col, lw=1.2))
        bar_x += 0.10

    # 12 Ableitungen
    # Plot nur erste 10 s (5000 Samples)
    t_plot = T[:5000]
    for lead_idx in range(12):
        ax = fig.add_subplot(gs[lead_idx + 1])
        ax.set_facecolor('#0d1117')
        sig = signal[:5000, lead_idx]
        # Thin gridlines
        ax.axhline(0, color=GRID_COLOR, linewidth=0.5)
        ax.plot(t_plot, sig, color=SIGNAL_COLOR, linewidth=0.6, rasterized=True)
        ax.set_xlim(0, 10)
        ax.set_ylabel(LEAD_NAMES[lead_idx], color='#8b949e', fontsize=8, rotation=0,
                      labelpad=20, va='center')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        # Amplitude range
        lo, hi = np.percentile(sig, 2), np.percentile(sig, 98)
        margin = max(0.1, (hi - lo) * 0.15)
        ax.set_ylim(lo - margin, hi + margin)

    # X-Achse nur unten
    axes = fig.axes
    axes[-1].tick_params(labelbottom=True, bottom=True, labelsize=7, colors='#8b949e')
    axes[-1].set_xlabel("Zeit (s)", color='#8b949e', fontsize=8)

    plt.savefig(filename, dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

# Alle 20 EKGs plotten
for rank, idx in enumerate(top20_idx, 1):
    ecg_id   = Y_test_meta.index[idx]
    gt_labels= [CLASS_NAMES[j] for j in range(len(CLASS_NAMES)) if y_test[idx,j]==1]
    pred_probs_dict = {CLASS_NAMES[j]: float(y_pred[idx,j]) for j in range(len(CLASS_NAMES))}
    bce      = float(bce_per_sample[idx])
    signal   = X_test[idx]   # Rohsignal (unnormalisiert)

    fname = f"{PLOTFOLDER}rank{rank:02d}_ecgid{ecg_id}.png"
    plot_ecg(signal, ecg_id, gt_labels, pred_probs_dict, rank, bce, fname)
    print(f"  #{rank:2d} gespeichert: {fname}")

# ─── 8. Übersichts-Plot (alle 20 Thumbnails) ───────────────────────────────────
print("\n=== Erstelle Übersichts-Plot (4×5) ===")
fig, axes = plt.subplots(4, 5, figsize=(22, 16))
fig.patch.set_facecolor('#0d1117')
fig.suptitle("Top-20 Fehlklassifikationen – 1D-ResNet auf PTB-XL (Fold 10, 500 Hz)\n"
             "Farbe = Lead II | Grün=TP  Orange=FN  Rot=FP  Grau=TN",
             color='#e6edf3', fontsize=13, fontweight='bold', y=0.99)

GT_COLOR = '#3fb950'; FN_COLOR = '#d29922'; FP_COLOR = '#f85149'; TN_COLOR = '#484f58'
SIGNAL_COLORS = ['#58a6ff', '#79c0ff', '#a5d6ff']

for ax_idx, (rank, idx) in enumerate(zip(range(1, 21), top20_idx)):
    ax = axes[ax_idx // 5][ax_idx % 5]
    ax.set_facecolor('#161b22')

    ecg_id     = Y_test_meta.index[idx]
    gt_set     = {CLASS_NAMES[j] for j in range(len(CLASS_NAMES)) if y_test[idx,j]==1}
    pred_set   = {CLASS_NAMES[j] for j in range(len(CLASS_NAMES)) if y_pred[idx,j]>=0.5}
    n_labels   = int(y_test[idx].sum())

    # Lead II als Thumbnail
    sig = X_test[idx][:2500, 1]   # erste 5 s
    t   = np.arange(len(sig)) / FS
    ax.plot(t, sig, color='#58a6ff', linewidth=0.7, rasterized=True)

    # Farbige Klassen-Tags
    tag_x = 0.02
    for cls in CLASS_NAMES:
        p = y_pred[idx, CLASS_NAMES.index(cls)]
        in_gt   = cls in gt_set
        in_pred = cls in pred_set
        col = GT_COLOR if (in_gt and in_pred) else \
              FN_COLOR if (in_gt and not in_pred) else \
              FP_COLOR if (not in_gt and in_pred) else TN_COLOR
        ax.text(tag_x, 0.04, cls, transform=ax.transAxes,
                color=col, fontsize=5.5, fontweight='bold', va='bottom')
        tag_x += 0.20

    # Multi-Label Marker
    label_info = f"{n_labels}×GT"
    ax.set_title(f"#{rank}  id={ecg_id}  {label_info}\nBCE={bce_per_sample[idx]:.3f}",
                 color='#e6edf3', fontsize=7, pad=3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#21262d')
    lo, hi = np.percentile(sig, 2), np.percentile(sig, 98)
    m = max(0.05, (hi-lo)*0.15)
    ax.set_ylim(lo-m, hi+m)

plt.tight_layout(rect=[0, 0, 1, 0.97])
overview_path = OUTFOLDER + "top20_errors_overview.png"
plt.savefig(overview_path, dpi=130, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f"Übersichts-Plot gespeichert: {overview_path}")

# ─── 9. Statistische Zusammenfassung ───────────────────────────────────────────
print("\n" + "="*65)
print("  FEHLER-MUSTER-ANALYSE – Top-20 am schlechtesten klassifiziert")
print("="*65)

# Multi-Label
ml_count = sum(1 for i in top20_idx if y_test[i].sum() >= 2)
print(f"\n1. Multi-Label-EKGs (≥2 Superklassen):  {ml_count}/20  ({ml_count*5:.0f}%)")

# Klassen-Bias
print(f"\n2. GT-Label-Häufigkeit in Top-20:")
for cls, cnt in Counter(all_gt).most_common():
    bar = '█' * cnt
    print(f"   {cls:<6}: {cnt:2d}x  {bar}")

# FN / FP Analyse
fn_count = {c:0 for c in CLASS_NAMES}
fp_count = {c:0 for c in CLASS_NAMES}
for idx in top20_idx:
    for j, cls in enumerate(CLASS_NAMES):
        gt_j   = y_test[idx, j]
        pred_j = y_pred[idx, j] >= 0.5
        if gt_j == 1 and not pred_j: fn_count[cls] += 1
        if gt_j == 0 and pred_j:     fp_count[cls] += 1

print(f"\n3. False Negatives (verpasste Klasse) in Top-20:")
for cls in CLASS_NAMES:
    if fn_count[cls]: print(f"   {cls}: {fn_count[cls]}x übersehen")

print(f"\n4. False Positives (falsch alarm) in Top-20:")
for cls in CLASS_NAMES:
    if fp_count[cls]: print(f"   {cls}: {fp_count[cls]}x fälschlich vorhergesagt")

# SCP-Code Häufigkeit (granulare Diagnosen)
all_scp = []
for idx in top20_idx:
    ecg_id = Y_test_meta.index[idx]
    all_scp.extend(list(Y_test_meta.loc[ecg_id].scp_codes.keys()))
print(f"\n5. Häufigste SCP-Codes in Top-20 (Granular-Diagnosen):")
for code, cnt in Counter(all_scp).most_common(15):
    desc = scp_df.loc[code, 'description'] if code in scp_df.index else "?"
    print(f"   {code:<8}: {cnt}x  – {desc}")

print(f"\n→ Einzelplots: {PLOTFOLDER}")
print(f"→ Übersicht:   {overview_path}")
