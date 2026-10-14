"""
Stufe 2: LLM-Befundgenerierung aus strukturiertem Faktum-Set
- Lädt example_fact_sets.json (Output von build_fact_set.py)
- Sendet jedes Faktum-Set an Claude claude-opus-4-5 mit klinischem System-Prompt
- Führt Konsistenzcheck durch: Zahlen im Text vs. Faktum-Set
- Speichert Ergebnisse in befund_results.json

Aufruf: ANTHROPIC_API_KEY=sk-ant-... python3 llm_befund.py
"""

import json
import os
import re
import sys
import anthropic

from ekg_config import BASE_STR  # zentrale Pfadkonfiguration


# ─── Konfiguration ────────────────────────────────────────────────────────────
FACTSET_FILE = BASE_STR + "example_fact_sets.json"
OUTFILE      = BASE_STR + "befund_results.json"
MODEL        = "claude-opus-4-5"

# ─── Klassen-Erklärungen für den Prompt ───────────────────────────────────────
CLASS_EXPLANATIONS = {
    "NORM": "Normalbefund",
    "MI":   "Myokardinfarkt (akut oder abgelaufen)",
    "STTC": "ST-Strecken-/T-Wellen-Veränderungen",
    "CD":   "Erregungsleitungsstörung (z. B. Schenkelblock, AV-Block)",
    "HYP":  "Hypertrophie (ventrikuläre oder atriale)",
}

REFERENZWERTE = {
    "hr_bpm":          (60, 100,  "bpm",  "Herzfrequenz"),
    "pr_interval_ms":  (120, 200, "ms",   "PQ-Intervall"),
    "qrs_duration_ms": (60,  120, "ms",   "QRS-Dauer"),
    "qt_interval_ms":  (350, 440, "ms",   "QT-Intervall"),
    "qtc_bazett_ms":   (None, 440,"ms",   "QTc (Bazett)"),
}

# ─── System-Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Du bist ein erfahrener Kardiologe und erstellst strukturierte EKG-Befunde für ein automatisiertes medizinisches Dokumentationssystem.

Du erhältst ein strukturiertes Faktum-Set mit:
- Patientendaten (Alter, Geschlecht)
- Klassische EKG-Messparameter (Herzfrequenz, Intervalle)
- KI-Klassifikationswahrscheinlichkeiten für 5 diagnostische Kategorien (NORM, MI, STTC, CD, HYP)
- Vorhergesagte Diagnosen (Schwellenwert: p ≥ 0.50)

Erstelle daraus einen strukturierten EKG-Befund im folgenden JSON-Format:

{
  "rhythmus": "<Sinusrhythmus/Vorhofflimmern/etc. mit Frequenzangabe>",
  "lagetyp": "<Normaltyp/Linkstyp/etc. — falls ableitbar, sonst 'nicht beurteilbar'>",
  "intervalle": {
    "pq": "<Wert in ms> — <normal/verlängert/verkürzt>",
    "qrs": "<Wert in ms> — <normal/verbreitert>",
    "qt": "<Wert in ms>",
    "qtc": "<Wert in ms> — <normal/verlängert>"
  },
  "befund": "<Freitext, 2-4 Sätze: Beschreibung der wichtigsten Auffälligkeiten>",
  "diagnosen": ["<Diagnose 1>", "<Diagnose 2>"],
  "beurteilung": "<1-2 Sätze klinische Einordnung und Dringlichkeit>",
  "hinweis": "<optionaler Satz zur Limitation der KI-Analyse, falls relevant>"
}

Wichtige Regeln:
- Verwende AUSSCHLIESSLICH die Werte aus dem Faktum-Set. Erfinde keine Befunde.
- Nenne konkrete Messwerte mit Einheit (z. B. "QRS 167 ms — verbreitert").
- Diagnosen nur aus den predicted_classes — keine Spekulation über nicht vorhergesagte Klassen.
- Sprache: medizinisches Deutsch, präzise und knapp.
- Antworte NUR mit dem JSON-Objekt, kein Fließtext darum herum."""

# ─── Prompt für ein einzelnes EKG ─────────────────────────────────────────────
def build_user_prompt(fs: dict) -> str:
    pat  = fs["patient"]
    feat = fs["classical_features"]
    prob = fs["resnet_probabilities"]

    ref_lines = []
    for key, (lo, hi, unit, label) in REFERENZWERTE.items():
        val = feat.get(key)
        if val is None:
            continue
        flag = ""
        if lo and val < lo:
            flag = " ⚠ UNTER Referenz"
        elif hi and val > hi:
            flag = " ⚠ ÜBER Referenz"
        ref_lines.append(f"  {label}: {val} {unit}{flag}")

    prob_lines = "\n".join(
        f"  {cls} ({CLASS_EXPLANATIONS[cls]}): {p:.1%}"
        for cls, p in sorted(prob.items(), key=lambda x: -x[1])
    )

    preds = ", ".join(fs["predicted_classes"])
    multi = "Ja" if fs["multi_label"] else "Nein"

    return f"""=== EKG-Faktum-Set (ecg_id: {fs['ecg_id']}) ===

Patient:
  Alter: {pat['age_years']} Jahre
  Geschlecht: {pat['sex']}

Messparameter (500 Hz, 10 Sekunden, 12 Ableitungen):
{chr(10).join(ref_lines)}

KI-Klassifikation (ResNet1D, PTB-XL trainiert):
{prob_lines}

Vorhergesagte Diagnosen: {preds}
Multi-Label: {multi}
Konfidenz: {fs['confidence']}

Erstelle jetzt den strukturierten EKG-Befund als JSON."""

# ─── Konsistenzcheck ──────────────────────────────────────────────────────────
def consistency_check(fs: dict, befund_text: str) -> list[dict]:
    """Prüft ob Zahlenwerte im Befundtext mit dem Faktum-Set übereinstimmen."""
    issues = []
    feat = fs["classical_features"]

    checks = {
        "hr_bpm":          (feat.get("hr_bpm"),          r"\b(\d{2,3}(?:\.\d)?)\s*(?:bpm|/min|Schläge)"),
        "qrs_duration_ms": (feat.get("qrs_duration_ms"), r"\bQRS[^0-9]*?(\d{2,3}(?:\.\d)?)\s*ms"),
        "pr_interval_ms":  (feat.get("pr_interval_ms"),  r"\bP[QR][^0-9]*?(\d{2,3}(?:\.\d)?)\s*ms"),
        "qtc_bazett_ms":   (feat.get("qtc_bazett_ms"),   r"\bQTc[^0-9]*?(\d{3,4}(?:\.\d)?)\s*ms"),
    }

    for param, (true_val, pattern) in checks.items():
        if true_val is None:
            continue
        matches = re.findall(pattern, befund_text, re.IGNORECASE)
        for m in matches:
            found = float(m)
            if abs(found - true_val) > 5:
                issues.append({
                    "parameter": param,
                    "expected":  true_val,
                    "found_in_text": found,
                    "delta": round(found - true_val, 1),
                })

    # Diagnosen-Check: predicted_classes müssen im Befund erwähnt sein
    diag_keywords = {
        "MI":   ["infarkt", "mi ", "ischä"],
        "CD":   ["leitungsstörung", "schenkelblock", "av-block", "block"],
        "HYP":  ["hypertrophie", "hyp"],
        "STTC": ["st-", "t-wellen", "t-veränderung", "sttc"],
        "NORM": ["normalbefund", "unauffällig", "regelrecht", "norm"],
    }
    text_lower = befund_text.lower()
    for cls in fs["predicted_classes"]:
        keywords = diag_keywords.get(cls, [])
        if not any(kw in text_lower for kw in keywords):
            issues.append({
                "parameter": f"diagnose_{cls}",
                "expected":  f"Erwähnung von {cls} ({CLASS_EXPLANATIONS[cls]})",
                "found_in_text": "nicht gefunden",
                "delta": None,
            })

    return issues

# ─── Hauptprogramm ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("FEHLER: ANTHROPIC_API_KEY nicht gesetzt.")
        print("Aufruf: ANTHROPIC_API_KEY=sk-ant-... python3 llm_befund.py")
        print("API Key erstellen: https://console.anthropic.com/settings/keys")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"=== Lade Faktum-Sets aus {FACTSET_FILE} ===")
    with open(FACTSET_FILE, encoding="utf-8") as f:
        fact_sets = json.load(f)

    results = []
    for fs in fact_sets:
        ecg_id = fs["ecg_id"]
        print(f"\n=== EKG {ecg_id} | true: {fs['true_classes']} | pred: {fs['predicted_classes']} ===")

        user_prompt = build_user_prompt(fs)

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_output = response.content[0].text
        usage      = response.usage

        # JSON aus Antwort extrahieren (Claude umschließt es manchmal mit ```json)
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        json_str   = json_match.group(0) if json_match else raw_output

        try:
            befund   = json.loads(json_str)
            parse_ok = True
        except json.JSONDecodeError as e:
            befund   = {"parse_error": str(e), "raw": raw_output}
            parse_ok = False

        issues = consistency_check(fs, raw_output) if parse_ok else []

        result = {
            "ecg_id":            ecg_id,
            "true_classes":      fs["true_classes"],
            "predicted_classes": fs["predicted_classes"],
            "befund":            befund,
            "konsistenz": {
                "ok":     len(issues) == 0,
                "issues": issues,
            },
            "tokens": {
                "input":  usage.input_tokens,
                "output": usage.output_tokens,
                "total":  usage.input_tokens + usage.output_tokens,
            },
        }
        results.append(result)

        print(json.dumps(befund, indent=2, ensure_ascii=False))
        if issues:
            print(f"\n⚠ Konsistenz-Issues ({len(issues)}):")
            for iss in issues:
                print(f"  {iss}")
        else:
            print("\n✓ Konsistenzcheck bestanden")

        total = usage.input_tokens + usage.output_tokens
        print(f"  Tokens: {usage.input_tokens} input + {usage.output_tokens} output = {total}")

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_tokens = sum(r["tokens"]["total"] for r in results)
    print(f"\n=== Ergebnisse gespeichert: {OUTFILE} ===")
    print(f"=== Gesamt-Tokens: {total_tokens} ===")
