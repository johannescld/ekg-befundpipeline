"""
Stufe 2, Version 2: Befundgenerierung nach dem überarbeiteten Schema.

Unterschiede zu llm_befund.py:
  - Prompt aus befund_schema.py (lehrbuchkonformer Aufbau, keine Prozentwerte)
  - Eingang ist das erweiterte Faktum-Set (build_fact_set_v2.py)
  - Konsistenzcheck erweitert um die neue Kernregel: der Befundtext darf
    keinerlei Modell- oder Wahrscheinlichkeitssprache enthalten
  - Zahlen-Regex gehärtet (bekannte Schwäche der Version 1)

Ohne API-Schlüssel schreibt das Skript die fertigen Prompts in eine Datei,
statt einen Lauf vorzutäuschen — ein in der Sitzung erzeugter Befund ist
kein Messergebnis und wird als solcher gekennzeichnet.

Aufruf:
    ANTHROPIC_API_KEY=sk-ant-... python3 llm_befund_v2.py      # echter Lauf
    python3 llm_befund_v2.py --dry-run                          # nur Prompts
"""

import argparse
import json
import os
import re
import sys

from befund_schema import (SYSTEM_PROMPT, BEFUND_SCHEMA, build_user_prompt,

                           render_befund)

from ekg_config import BASE_STR  # zentrale Pfadkonfiguration

BASE = BASE_STR
FACTSET_FILE = BASE + "example_fact_sets_v2.json"
OUTFILE = BASE + "befund_results_v2.json"
PROMPT_DUMP = BASE + "befund_prompts_v2.txt"
MODEL = "claude-opus-4-5"      # beim echten Lauf auf die aktuelle Version setzen



# ─── Modellaufruf: lokal oder extern ─────────────────────────────────────────
def frage_modell(system: str, user: str, bild_b64: str = None) -> tuple:
    """
    Schickt eine Anfrage an das Sprachmodell und gibt (Text, Tokenzahlen) zurück.

    Zwei Wege, in dieser Reihenfolge geprüft:

    1. LOKAL — wenn EKG_LLM_ENDPOINT gesetzt ist. Erwartet eine
       OpenAI-kompatible Schnittstelle, wie sie Ollama (Standardport 11434)
       und vLLM (8000) bereitstellen. Die Daten verlassen den Rechner nicht.

           export EKG_LLM_ENDPOINT=http://localhost:11434/v1
           export EKG_LLM_MODEL=llama3.1:8b

    2. EXTERN — wenn stattdessen ANTHROPIC_API_KEY gesetzt ist.
       **Für Patientendaten nicht vorgesehen.** Dieser Weg diente der
       Entwicklung auf dem öffentlichen Datensatz.

    Ist keines von beiden gesetzt, wird nichts aufgerufen.
    """
    endpunkt = os.environ.get("EKG_LLM_ENDPOINT")
    if endpunkt:
        import json as _json
        import urllib.request
        modell = os.environ.get("EKG_LLM_MODEL", "llama3.1:8b")
        inhalt = user if not bild_b64 else [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{bild_b64}"}},
            {"type": "text", "text": user}]
        anfrage = {
            "model": modell,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": inhalt}],
            "temperature": 0.2,
            "max_tokens": 1500,
        }
        req = urllib.request.Request(
            endpunkt.rstrip("/") + "/chat/completions",
            data=_json.dumps(anfrage).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as antwort:
                daten = _json.loads(antwort.read())
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Das lokale Sprachmodell ist unter {endpunkt} nicht "
                f"erreichbar ({e.reason}).\n"
                f"  - Läuft der Dienst?   Ollama:  ollama serve\n"
                f"                        vLLM:    python -m vllm.entrypoints."
                f"openai.api_server --model <modell>\n"
                f"  - Ist das Modell geladen?  ollama pull "
                f"{os.environ.get('EKG_LLM_MODEL', 'llama3.1:8b')}\n"
                f"  - Stimmt der Port? Ollama nutzt 11434, vLLM 8000.") from None
        if "choices" not in daten:
            raise RuntimeError(
                f"Unerwartete Antwort des lokalen Modells: "
                f"{str(daten)[:200]}\nErwartet wird eine OpenAI-kompatible "
                f"Schnittstelle.")
        nutzung = daten.get("usage", {})
        return (daten["choices"][0]["message"]["content"],
                {"input": nutzung.get("prompt_tokens"),
                 "output": nutzung.get("completion_tokens"),
                 "modell": modell, "weg": "lokal"})

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        inhalt = user if not bild_b64 else [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": bild_b64}},
            {"type": "text", "text": user}]
        antwort = client.messages.create(
            model=MODEL, max_tokens=1500, temperature=0.2,
            system=system, messages=[{"role": "user", "content": inhalt}])
        return (antwort.content[0].text,
                {"input": antwort.usage.input_tokens,
                 "output": antwort.usage.output_tokens,
                 "modell": MODEL, "weg": "extern"})

    raise RuntimeError(
        "Kein Sprachmodell konfiguriert. Entweder EKG_LLM_ENDPOINT auf eine "
        "lokale OpenAI-kompatible Schnittstelle setzen (empfohlen, bleibt im "
        "Haus) oder ANTHROPIC_API_KEY für den externen Dienst.")


# ─── Konsistenzcheck ─────────────────────────────────────────────────────────
# Verbotene Muster: alles, was den Befund als Maschinenausgabe entlarvt.
VERBOTENE_MUSTER = [
    (r"\d{1,3}\s*%", "Prozentangabe im Befundtext"),
    (r"\bKI\b|künstliche[rn]? Intelligenz", "Nennung der KI"),
    (r"\bModell\b|\bAlgorithmus\b|\bKlassifikat", "Nennung des Modells"),
    (r"\bWahrscheinlichkeit von\b|\bKonfidenz\b", "Wahrscheinlichkeitssprache"),
    (r"\bResNet\b|\bneuronale[sn]?\b", "Nennung der Architektur"),
    (r"\bvorhergesagt\b|\bPrädiktion\b", "Vorhersagesprache"),
]

# Zahlenprüfung: Distanz zwischen Parametername und Zahl begrenzt, sonst
# greift der Ausdruck auf weit entfernte Zahlen zu (Fehler der Version 1,
# dort stand [^0-9]*? ohne Obergrenze).
ZAHL_MUSTER = {
    "hr_bpm":          r"\b(\d{2,3}(?:[.,]\d)?)\s*(?:/min|bpm|Schläge)",
    "qrs_duration_ms": r"\bQRS[^0-9]{0,15}?(\d{2,3}(?:[.,]\d)?)\s*ms",
    "pr_interval_ms":  r"\bPQ[^0-9]{0,15}?(\d{2,3}(?:[.,]\d)?)\s*ms",
    "qtc_bazett_ms":   r"\bQTc[^0-9]{0,15}?(\d{3}(?:[.,]\d)?)\s*ms",
}


def konsistenzcheck(fs: dict, text: str) -> list:
    """Prüft Befundtext gegen Faktum-Set und gegen die Stilregeln."""
    issues = []

    # 1. Stilregeln — kein Modell-Sprech
    for muster, beschreibung in VERBOTENE_MUSTER:
        for treffer in re.findall(muster, text, re.IGNORECASE):
            issues.append({"typ": "stil", "regel": beschreibung,
                           "fundstelle": str(treffer)})

    # 2. Zahlenwerte gegen das Faktum-Set (nur verwertbare Messwerte)
    feats = fs.get("classical_features_verwertbar") or \
        fs.get("classical_features", {})
    for param, muster in ZAHL_MUSTER.items():
        soll = feats.get(param)
        if soll is None:
            continue
        for m in re.findall(muster, text, re.IGNORECASE):
            ist = float(str(m).replace(",", "."))
            if abs(ist - float(soll)) > 5:
                issues.append({"typ": "zahl", "parameter": param,
                               "erwartet": soll, "im_text": ist,
                               "differenz": round(ist - float(soll), 1)})

    # 3. Als Artefakt verworfene Messwerte dürfen nicht als Wert auftauchen
    for art in fs.get("plausibility", {}).get("artefakte", []):
        name = art.split()[0] if isinstance(art, str) else ""
        if name and re.search(rf"\b{re.escape(name)}\b[^.]*\d", text):
            issues.append({"typ": "artefakt", "parameter": name,
                           "hinweis": "verworfener Messwert erscheint mit Zahl"})

    # 4. Jede gemeldete Diagnose muss in der Beurteilung vorkommen
    for d in fs.get("diagnose", {}).get("gemeldet", []):
        kern = (d.get("bezeichnung") or "").split(",")[0].split("/")[0]
        kern = kern.replace("unspezifische ", "").strip()
        if kern and len(kern) > 4 and kern.lower() not in text.lower():
            issues.append({"typ": "diagnose", "erwartet": d.get("bezeichnung"),
                           "hinweis": "nicht im Befundtext erwähnt"})
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="nur Prompts schreiben, keinen API-Lauf versuchen")
    ap.add_argument("--factsets", default=FACTSET_FILE)
    args = ap.parse_args()

    with open(args.factsets, encoding="utf-8") as f:
        fact_sets = json.load(f)

    konfiguriert = bool(os.environ.get("EKG_LLM_ENDPOINT")
                        or os.environ.get("ANTHROPIC_API_KEY"))

    # ── Ohne Modell: Prompts dokumentieren, nichts erfinden ─────────────────
    if args.dry_run or not konfiguriert:
        with open(PROMPT_DUMP, "w", encoding="utf-8") as f:
            f.write("SYSTEM-PROMPT\n" + "=" * 70 + "\n")
            f.write(SYSTEM_PROMPT + "\n\n")
            f.write("SCHEMA\n" + "=" * 70 + "\n")
            f.write(json.dumps(BEFUND_SCHEMA, indent=2, ensure_ascii=False))
            f.write("\n\n")
            for fs in fact_sets:
                f.write("=" * 70 + f"\nUSER-PROMPT — EKG {fs['ecg_id']}\n"
                        + "=" * 70 + "\n")
                f.write(build_user_prompt(fs) + "\n\n")
        print(f"Prompts geschrieben: {PROMPT_DUMP}")
        if not konfiguriert:
            print("\nKein Sprachmodell konfiguriert — es wurde KEIN Befund "
                  "erzeugt.\nEin in einer Sitzung formulierter Befund wäre kein "
                  "reproduzierbares Messergebnis.\n")
            print("Lokales Modell (empfohlen, Daten bleiben im Haus):")
            print("  export EKG_LLM_ENDPOINT=http://localhost:11434/v1")
            print("  export EKG_LLM_MODEL=llama3.1:8b")
            print("  python3 llm_befund_v2.py\n")
            print("Externer Dienst (nur für öffentliche Daten):")
            print("  ANTHROPIC_API_KEY=sk-ant-... python3 llm_befund_v2.py")
        return

    # ── Lauf ───────────────────────────────────────────────────────────────
    results = []
    for fs in fact_sets:
        print(f"\n=== EKG {fs['ecg_id']} (wahr: {fs.get('true_classes')}) ===")
        raw, nutzung = frage_modell(SYSTEM_PROMPT, build_user_prompt(fs))
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            befund = json.loads(m.group(0) if m else raw)
            parse_ok = True
        except json.JSONDecodeError as e:
            befund, parse_ok = {"parse_error": str(e), "raw": raw}, False

        text = render_befund(befund) if parse_ok else raw
        issues = konsistenzcheck(fs, text)

        results.append({
            "ecg_id": fs["ecg_id"], "modell": nutzung.get("modell"),
            "weg": nutzung.get("weg"),
            "true_classes": fs.get("true_classes"),
            "gemeldete_diagnosen": fs["diagnose"]["gemeldet"],
            "befund": befund, "befundtext": text,
            "konsistenz": {"ok": not issues, "issues": issues},
            "tokens": {"input": nutzung.get("input"),
                       "output": nutzung.get("output")},
        })

        print(text)
        print("\n✓ Konsistenzcheck bestanden" if not issues
              else f"\n⚠ {len(issues)} Beanstandung(en): {issues}")

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    n_ok = sum(1 for r in results if r["konsistenz"]["ok"])
    print(f"\n=== {n_ok}/{len(results)} ohne Beanstandung — {OUTFILE} ===")


if __name__ == "__main__":
    main()
