"""
Hybrides Label-Schema für Stufe 1 (Re-Training).

Verfeinert die 5 PTB-XL-Superklassen auf 19 Zielklassen. Leitlinie ist die
Tragfähigkeit der Daten: eine Subklasse bleibt eigenständig, wenn sie
mindestens 20 Fälle im Test-Fold 10 aufweist; seltenere werden je
Superklasse zu einem '<SUPER>_other'-Bucket aggregiert.

Diese 20er-Marke ist ein RICHTWERT, kein Automatismus — das Schema weicht
an zwei Stellen bewusst von ihr ab. Beide Abweichungen sind unten am
jeweiligen Mapping-Eintrag begründet; zusammengefasst:

  LMI (20 Testfälle) wird trotz erfüllter Marke gebündelt, weil PMI mit
  17 Fällen (2 im Test) sonst als eigene Restklasse zurückbliebe.

  CD_other (16 Testfälle) wird trotz verfehlter Marke geführt, weil sonst
  74 EKGs ihre einzige Zielklasse verlören und aus dem Datensatz fielen.

Eine strikte Anwendung der Marke ergäbe 18 Zielklassen. Die empirische
Folge der Abweichung bei CD_other ist ausgewiesen: Bootstrap-95%-KI der
AUROC [0.693-0.983] — die schwächste Klasse des Schemas (s.
ptbxl_results/class_metrics_hybrid.csv, class_granularity.py).

Verwendung im Trainingsskript:
    from label_mapping_hybrid import build_label_matrix, TARGET_CLASSES
    mlb, Y_bin = build_label_matrix(Y)        # Y = ptbxl_database.csv (scp_codes geparst)
    # Y_bin: (n_ekg, 19) Multi-Hot, Reihenfolge = mlb.classes_

Standalone-Aufruf (Verifikation der Fold-Größen):
    python3 label_mapping_hybrid.py
"""

import ast
import numpy as np
import pandas as pd

from ekg_config import BASE_STR  # zentrale Pfadkonfiguration


DATAFOLDER = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"

# ─── Hybrides Mapping: Subklasse -> finale Zielklasse ─────────────────────────
# Identität = eigene Klasse (genug Daten); '*_other' = aggregierter Rest-Bucket.
SUBCLASS_TO_TARGET = {
    # NORM
    "NORM":      "NORM",
    # MI — anterior/inferior eigenständig; lateral+posterior gebündelt
    "AMI":       "AMI",          # Vorderwandinfarkt
    "IMI":       "IMI",          # Hinterwand-/Inferiorinfarkt
    # ABWEICHUNG 1 vom 20er-Richtwert: LMI erreicht mit genau 20 Testfällen
    # die Marke, wird aber dennoch gebündelt. Eigenständig geführt bliebe PMI
    # (17 Fälle gesamt, 2 im Test) als nicht tragfähige Restklasse zurück —
    # oder 5 EKGs mit PMI als einziger Diagnose müssten verworfen werden.
    # Klinisch bilden Lateral- und Posteriorinfarkt zusammen die Kategorie
    # "weder anterior noch inferior", die Bündelung ist daher auch
    # inhaltlich eine Einheit und keine reine Verlegenheitslösung.
    "LMI":       "MI_other",     # Lateralinfarkt   } gemeinsam
    "PMI":       "MI_other",     # Posteriorinfarkt } als Einheit
    # STTC — alle 5 Subklassen tragfähig
    "STTC":      "STTC",         # unspezifische ST-T-Veränderungen
    "NST_":      "NST_",         # unspezifische ST-Strecken-Veränderungen
    "ISC_":      "ISC_",         # unspezifische Ischämie
    "ISCA":      "ISCA",         # Ischämie anterior
    "ISCI":      "ISCI",         # Ischämie inferior
    # CD — 6 tragfähig; WPW+ILBBB selten -> gebündelt
    "LAFB/LPFB": "LAFB/LPFB",    # Hemiblock
    "IRBBB":     "IRBBB",        # inkompletter RSB
    "CRBBB":     "CRBBB",        # kompletter RSB
    "CLBBB":     "CLBBB",        # kompletter LSB
    "_AVB":      "_AVB",         # AV-Block
    "IVCD":      "IVCD",         # intraventrikuläre Leitungsverzögerung
    # ABWEICHUNG 2 vom 20er-Richtwert: CD_other erreicht mit 16 Testfällen
    # die Marke nicht, wird aber dennoch als Klasse geführt. Andernfalls
    # verlören 74 EKGs (9 davon in Fold 10) ihre einzige Zielklasse und
    # fielen beim Filterschritt aus dem Datensatz — darunter sämtliche
    # reinen WPW-Aufzeichnungen. Der Preis ist eine statistisch nicht
    # belastbare Klasse (AUROC-KI [0.693-0.983]); sie ist im Ergebnisteil
    # entsprechend zu kennzeichnen und nicht zu interpretieren.
    "WPW":       "CD_other",     # Präexzitation
    "ILBBB":     "CD_other",     # inkompletter LSB
    # HYP — nur LVH + LAO/LAE tragfähig; Rest gebündelt
    "LVH":       "LVH",          # linksventrikuläre Hypertrophie
    "LAO/LAE":   "LAO/LAE",      # linksatriale Belastung
    "RVH":       "HYP_other",    # rechtsventr. Hypertrophie }
    "RAO/RAE":   "HYP_other",    # rechtsatriale Belastung   } selten -> gebündelt
    "SEHYP":     "HYP_other",    # Septumhypertrophie        }
}

# Zielklasse -> Superklasse (für Auswertung / Fallback in Stufe 2/3)
TARGET_TO_SUPER = {
    "NORM": "NORM",
    "AMI": "MI", "IMI": "MI", "MI_other": "MI",
    "STTC": "STTC", "NST_": "STTC", "ISC_": "STTC", "ISCA": "STTC", "ISCI": "STTC",
    "LAFB/LPFB": "CD", "IRBBB": "CD", "CRBBB": "CD", "CLBBB": "CD",
    "_AVB": "CD", "IVCD": "CD", "CD_other": "CD",
    "LVH": "HYP", "LAO/LAE": "HYP", "HYP_other": "HYP",
}

# Deutschsprachige Klartextbezeichnung (für Befundtext in Stufe 2)
TARGET_LABELS_DE = {
    "NORM":      "Normalbefund",
    "AMI":       "Vorderwandinfarkt",
    "IMI":       "Hinterwand-/Inferiorinfarkt",
    "MI_other":  "Myokardinfarkt, lateral/posterior",
    "STTC":      "unspezifische ST-T-Veränderungen",
    "NST_":      "unspezifische ST-Strecken-Veränderungen",
    "ISC_":      "Ischämie, unspezifisch",
    "ISCA":      "Ischämie, anteriore Ableitungen",
    "ISCI":      "Ischämie, inferiore Ableitungen",
    "LAFB/LPFB": "linksanteriorer/-posteriorer Hemiblock",
    "IRBBB":     "inkompletter Rechtsschenkelblock",
    "CRBBB":     "kompletter Rechtsschenkelblock",
    "CLBBB":     "kompletter Linksschenkelblock",
    "_AVB":      "AV-Block",
    "IVCD":      "intraventrikuläre Leitungsverzögerung",
    "CD_other":  "Erregungsleitungsstörung, sonstige (WPW/inkompl. LSB)",
    "LVH":       "linksventrikuläre Hypertrophie",
    "LAO/LAE":   "linksatriale Belastung/Vergrößerung",
    "HYP_other": "Hypertrophie/Belastung, rechtsseitig/septal",
}

# Finale Klassenliste (alphabetisch sortiert = Konvention MultiLabelBinarizer)
TARGET_CLASSES = sorted(set(SUBCLASS_TO_TARGET.values()))


# ─── Aufbau der SCP-Code -> Zielklasse-Auflösung (über scp_statements.csv) ─────
def _build_code_to_target(datafolder: str = DATAFOLDER) -> dict:
    """SCP-Code -> Zielklasse, abgeleitet über die diagnostische Subklasse."""
    scp = pd.read_csv(datafolder + "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1.0]
    code_to_target = {}
    for code, row in diag.iterrows():
        sub = row.diagnostic_subclass
        if pd.notna(sub) and sub in SUBCLASS_TO_TARGET:
            code_to_target[code] = SUBCLASS_TO_TARGET[sub]
    return code_to_target


def ecg_to_targets(scp_codes: dict, code_to_target: dict) -> list:
    """Set der Zielklassen für ein EKG (Multi-Label). scp_codes = {code: likelihood}."""
    return sorted({code_to_target[c] for c in scp_codes if c in code_to_target})


def build_label_matrix(Y: pd.DataFrame, datafolder: str = DATAFOLDER):
    """
    Y: PTB-XL-Metadaten mit bereits geparster Spalte scp_codes (dict).
    Rückgabe: (mlb, Y_bin) — Y_bin Multi-Hot (n, 19), Spaltenreihenfolge mlb.classes_.
    EKGs ohne jede Zielklasse erhalten eine reine Nullzeile (im Training filterbar).
    """
    from sklearn.preprocessing import MultiLabelBinarizer
    code_to_target = _build_code_to_target(datafolder)
    target_sets = Y.scp_codes.apply(lambda d: ecg_to_targets(d, code_to_target))
    mlb = MultiLabelBinarizer(classes=TARGET_CLASSES)
    Y_bin = mlb.fit_transform(target_sets)
    return mlb, Y_bin


# ─── Verifikation ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from collections import Counter

    print(f"=== Hybrides Label-Schema: {len(TARGET_CLASSES)} Zielklassen ===")
    print(TARGET_CLASSES)

    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    code_to_target = _build_code_to_target()
    Y["targets"] = Y.scp_codes.apply(lambda d: ecg_to_targets(d, code_to_target))

    splits = {
        "Train(1-8)": Y[Y.strat_fold <= 8],
        "Val(9)":     Y[Y.strat_fold == 9],
        "Test(10)":   Y[Y.strat_fold == 10],
    }
    cnt = {n: Counter(t for ts in df.targets for t in ts) for n, df in splits.items()}

    n_empty = int((Y.targets.map(len) == 0).sum())
    print(f"\nEKGs ohne Zielklasse (nur Form/Rhythmus o. nicht-diagn.): {n_empty}")

    print(f"\n{'Zielklasse':11s} {'Super':5s} {'Train':>6s} {'Val':>4s} {'Test':>5s}  Status")
    print("-" * 60)
    for t in sorted(TARGET_CLASSES, key=lambda x: -cnt['Test(10)'].get(x, 0)):
        te = cnt["Test(10)"].get(t, 0)
        status = "ok" if te >= 20 else "GRENZWERTIG"
        print(f"{t:11s} {TARGET_TO_SUPER[t]:5s} "
              f"{cnt['Train(1-8)'].get(t,0):6d} {cnt['Val(9)'].get(t,0):4d} {te:5d}  {status}")

    weak = [t for t in TARGET_CLASSES if cnt["Test(10)"].get(t, 0) < 20]
    print(f"\n{len(TARGET_CLASSES) - len(weak)}/{len(TARGET_CLASSES)} Zielklassen mit >= 20 Testfällen.")
    if weak:
        print("Unterhalb des Richtwerts:", {t: cnt['Test(10)'].get(t, 0) for t in weak})
    print("\nHinweis: Der 20er-Richtwert wird an zwei Stellen bewusst verlassen")
    print("(LMI gebündelt trotz 20 Fällen, CD_other geführt trotz 16 Fällen).")
    print("Begründung siehe Modul-Docstring; strikt angewandt ergäben sich 18")
    print("statt 19 Zielklassen.")
