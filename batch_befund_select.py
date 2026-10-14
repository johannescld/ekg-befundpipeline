"""
Wählt pro Zielkategorie den selbstbewusst & korrekt klassifizierten Vertreter
aus dem Test-Fold (Hybrid-19-Modell) und baut die Faktum-Sets (inkl.
Plausibilität + Konfidenz-Gating) — als Beispiel-Material für die Doktorarbeit.

Output: batch_fact_sets.json
"""
import json, warnings, numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
from build_fact_set import resnet1d_wang, extract_classical_features, confidence_label, DATAFOLDER, SR
from plausibility import annotate_fact_set
from label_mapping_hybrid import build_label_matrix, TARGET_TO_SUPER, TARGET_LABELS_DE
from ekg_befund import get_norm_stats, befund_diagnosen, HYBRID_MODEL_PATH, THRESHOLD
from ekg_config import get_device

# Kuratiert: klare Einzel-Pathologien (Modellstärken) + 1 Multi-Label
TARGETS = ["CLBBB", "CRBBB", "AMI", "IMI", "LVH", "NORM"]

Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
import ast; Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
X = np.load(DATAFOLDER + "raw500_signals.npy", allow_pickle=False)
pos_map = {e: i for i, e in enumerate(Y.index)}

mlb, y_full = build_label_matrix(Y)
class_names = list(mlb.classes_)
test_ids = list(Y.index[Y.strat_fold == 10])

mean_tr, std_tr = get_norm_stats()
device = get_device()
model = resnet1d_wang(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(HYBRID_MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# Inferenz über Test-Fold
probs_by_id = {}
for eid in test_ids:
    s = (X[pos_map[eid]] - mean_tr[0]) / std_tr[0]
    with torch.no_grad():
        probs_by_id[eid] = torch.sigmoid(model(torch.FloatTensor(s.T[np.newaxis]).to(device))).cpu().numpy()[0]

def true_set(eid):
    return {c for c, v in zip(class_names, y_full[pos_map[eid]]) if v == 1}
def pred_set(p):
    s = {c for c, v in zip(class_names, p) if v >= THRESHOLD}
    return s or {class_names[int(np.argmax(p))]}

# Auswahl je Ziel: reine Einzel-Diagnose, exakt korrekt, höchste Konfidenz
selected = {}
for tgt in TARGETS:
    cands = [(eid, float(probs_by_id[eid][class_names.index(tgt)]))
             for eid in test_ids
             if true_set(eid) == {tgt} and pred_set(probs_by_id[eid]) == {tgt}]
    if cands:
        selected[tgt] = max(cands, key=lambda x: x[1])[0]

# 1 selbstbewusstes Multi-Label ergänzen
multi = [(eid, float(probs_by_id[eid].max())) for eid in test_ids
         if len(true_set(eid)) >= 2 and pred_set(probs_by_id[eid]) == true_set(eid)]
if multi:
    selected["MULTI"] = max(multi, key=lambda x: x[1])[0]

# Faktum-Sets bauen
fact_sets = []
for tgt, eid in selected.items():
    sig = X[pos_map[eid]]
    probs = probs_by_id[eid]
    meta = Y.loc[eid]
    fs = {
        "ecg_id": int(eid), "auswahl_kategorie": tgt, "schema": "hybrid_19",
        "patient": {"age_years": int(meta["age"]) if not pd.isna(meta["age"]) else None,
                    # PTB-XL: 0 = männlich, 1 = weiblich (vgl. build_fact_set.py)
                    "sex": "weiblich" if meta["sex"] == 1 else "männlich"},
        "acquisition": {"sampling_rate_hz": SR, "duration_s": 10.0, "n_leads": 12},
        "classical_features": extract_classical_features(sig, sr=SR),
        "resnet_probabilities": {c: round(float(p), 3) for c, p in zip(class_names, probs)},
        "predicted_classes": sorted(pred_set(probs)),
        "true_classes": sorted(true_set(eid)),
        "confidence": confidence_label(float(probs.max())),
        "pipeline_stage": "stufe_1_output",
    }
    fs = annotate_fact_set(fs)
    diag, gate = befund_diagnosen(class_names, probs, hybrid=True)
    fs["befund_diagnosen"] = diag
    if gate: fs["konfidenz_gating"] = gate
    fact_sets.append(fs)
    top = {k: v for k, v in sorted(fs["resnet_probabilities"].items(), key=lambda x: -x[1])[:4]}
    art = "" if fs["plausibility"]["alle_verwertbar"] else f" | ⚠{fs['plausibility']['n_artefakte']} Artefakt"
    print(f"{tgt:6s} EKG {eid:5d} | {fs['patient']['age_years']}J {fs['patient']['sex'][:1]} | "
          f"pred={fs['predicted_classes']} | top={top}{art}")
    print(f"        feats={ {k:(round(v,1) if isinstance(v,(int,float)) else v) for k,v in fs['classical_features'].items()} }")
    print(f"        Diagnose(n)={diag}")

json.dump(fact_sets, open("batch_fact_sets.json", "w"), indent=2, ensure_ascii=False)
print(f"\n=== {len(fact_sets)} Faktum-Sets gespeichert: batch_fact_sets.json ===")
