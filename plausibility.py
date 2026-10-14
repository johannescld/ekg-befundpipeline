"""
Stufe 3 (Keim): Plausibilitätsprüfung der gemessenen EKG-Intervalle.

Zweck: Automatische Delineation (neurokit2) verschätzt sich bei pathologischen
EKGs systematisch (z. B. PQ 52 ms bei EKG 526). Solche physiologisch unmöglichen
Werte dürfen NICHT als echter Befund ins LLM gelangen.

Zwei getrennte Ebenen — wichtig für die Doktorarbeit:
  - REFERENZ  = Normbereich (z. B. PQ 120-200 ms). Außerhalb = AUFFÄLLIG, aber real
                -> gehört in den Befund als pathologischer Befund.
  - PLAUSIBEL = physiologische Möglichkeit (z. B. PQ 80-350 ms). Außerhalb = ARTEFAKT
                -> Wert verwerfen, NICHT als Befund ausgeben.

check_features() liefert pro Messwert einen Status; annotate_fact_set() hängt
das Ergebnis ans Faktum-Set und markiert unverwertbare Werte.
"""

# (untere, obere) physiologische Plausibilitätsgrenze — außerhalb = Messartefakt
PLAUSIBILITY_BOUNDS = {
    "hr_bpm":          (25,  300),
    "pr_interval_ms":  (80,  350),
    "qrs_duration_ms": (40,  200),
    "qt_interval_ms":  (200, 700),
    "qtc_bazett_ms":   (300, 700),
}

# Normbereich (für die klinische Einordnung im Befund) — None = keine Grenze
REFERENZ = {
    "hr_bpm":          (60,  100),
    "pr_interval_ms":  (120, 200),
    "qrs_duration_ms": (60,  120),
    "qt_interval_ms":  (350, 440),
    "qtc_bazett_ms":   (None, 440),
}

LABELS = {
    "hr_bpm":          ("Herzfrequenz", "bpm"),
    "pr_interval_ms":  ("PQ-Intervall", "ms"),
    "qrs_duration_ms": ("QRS-Dauer",    "ms"),
    "qt_interval_ms":  ("QT-Intervall", "ms"),
    "qtc_bazett_ms":   ("QTc (Bazett)", "ms"),
}


def _ref_status(param, val):
    """innerhalb / unter / über Normbereich (nur für plausible Werte)."""
    lo, hi = REFERENZ.get(param, (None, None))
    if lo is not None and val < lo:
        return "unter_norm"
    if hi is not None and val > hi:
        return "ueber_norm"
    return "normal"


def check_features(feats: dict) -> dict:
    """
    Prüft jeden Messwert. Rückgabe pro Parameter:
      {value, plausibel: bool, verwertbar: bool, status, ref_status, kommentar}
    plus aggregierte Listen 'warnungen' und 'artefakte'.
    """
    result, warnungen, artefakte = {}, [], []
    for param, (lo, hi) in PLAUSIBILITY_BOUNDS.items():
        label, unit = LABELS[param]
        val = feats.get(param)

        if val is None:
            result[param] = {"value": None, "plausibel": False, "verwertbar": False,
                             "status": "fehlt", "ref_status": None,
                             "kommentar": f"{label} konnte nicht gemessen werden."}
            warnungen.append(f"{label}: nicht messbar.")
            continue

        if val < lo or val > hi:
            richtung = "unrealistisch niedrig" if val < lo else "unrealistisch hoch"
            kommentar = (f"{label} {val} {unit} ist {richtung} "
                         f"(plausibel {lo}–{hi} {unit}) — mutmaßliches Messartefakt, "
                         f"nicht als Befund verwertbar.")
            result[param] = {"value": val, "plausibel": False, "verwertbar": False,
                             "status": "artefakt", "ref_status": None, "kommentar": kommentar}
            artefakte.append(kommentar)
            continue

        ref = _ref_status(param, val)
        result[param] = {"value": val, "plausibel": True, "verwertbar": True,
                         "status": "ok", "ref_status": ref,
                         "kommentar": f"{label} {val} {unit} ({ref.replace('_', ' ')})."}
        if ref != "normal":
            warnungen.append(result[param]["kommentar"])

    return {
        "pro_parameter": result,
        "warnungen": warnungen,
        "artefakte": artefakte,
        "n_artefakte": len(artefakte),
        "alle_verwertbar": len(artefakte) == 0,
    }


def annotate_fact_set(fs: dict) -> dict:
    """Hängt 'plausibility' ans Faktum-Set und ergänzt verwertbare Features."""
    check = check_features(fs.get("classical_features", {}))
    fs["plausibility"] = check
    # verwertbare Feature-Variante: Artefakte auf None gesetzt
    fs["classical_features_verwertbar"] = {
        p: (info["value"] if info["verwertbar"] else None)
        for p, info in check["pro_parameter"].items()
    }
    return fs


if __name__ == "__main__":
    # Selbsttest an den bekannten Beispiel-EKGs
    import json, os
    path = os.path.join(os.path.dirname(__file__), "example_fact_sets.json")
    for fs in json.load(open(path, encoding="utf-8")):
        c = check_features(fs["classical_features"])
        tag = "✓ alle plausibel" if c["alle_verwertbar"] else f"⚠ {c['n_artefakte']} Artefakt(e)"
        print(f"EKG {fs['ecg_id']:5d}: {tag}")
        for a in c["artefakte"]:
            print(f"        → {a}")
