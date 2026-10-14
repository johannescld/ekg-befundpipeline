"""
Stufe 2 (lokal): LLM-Befundgenerierung OHNE Anthropic-API.

Die Befundtexte wurden direkt von Claude Opus 4.8 (in der Claude-Code-Session)
nach dem identischen System-Prompt aus llm_befund.py erzeugt und hier hinterlegt.
Wiederverwendet 1:1:
  - consistency_check aus llm_befund.py  (Zahl-im-Text vs. Faktum-Set)
  - check_features  aus plausibility.py  (Messartefakt-Erkennung, Stufe-3-Keim)

Faktum-Sets: example_fact_sets.json (korrigierte Signal-Indizierung, Stand 2026-06-30).
Aufruf: python3 llm_befund_local.py
"""

import json
from llm_befund import consistency_check, FACTSET_FILE, OUTFILE
from plausibility import check_features

BACKEND = "claude-opus-4-8 (lokal, Claude-Code-Session, kein API-Key)"

# ─── Von Claude (Kardiologe-Rolle, plausibilitäts-bewusst) generierte Befunde ──
BEFUNDE = {
    65: {
        "rhythmus": "Sinusrhythmus, normofrequent (72/min)",
        "lagetyp": "nicht beurteilbar",
        "intervalle": {"pq": "137 ms — normal", "qrs": "99 ms — normal",
                       "qt": "359 ms", "qtc": "391 ms — normal"},
        "befund": (
            "Normofrequenter Sinusrhythmus (72/min) mit regelrechter PQ-Zeit (137 ms) "
            "und schmalem QRS-Komplex (99 ms). Die KI-Analyse klassifiziert mit hoher "
            "Wahrscheinlichkeit eine Erregungsleitungsstörung (CD, 97 %); ein Normalbefund "
            "liegt mit 48 % als Differenzialdiagnose nahe der Entscheidungsschwelle. Bei "
            "normaler Kammerkomplexbreite am ehesten faszikuläre oder AV-nodale Störung, "
            "kein kompletter Schenkelblock."
        ),
        "diagnosen": ["Erregungsleitungsstörung (z. B. Schenkelblock, AV-Block)"],
        "beurteilung": (
            "Wahrscheinliche Erregungsleitungsstörung bei hoher Konfidenz, jedoch knapper "
            "Abgrenzung zum Normalbefund; ärztliche Korrelation mit der 12-Kanal-Morphologie "
            "empfohlen."
        ),
        "hinweis": "Bei schmalem Kammerkomplex ist eine höhergradige Blockierung unwahrscheinlich.",
    },
    7670: {
        "rhythmus": "Sinusrhythmus, normofrequent (80/min)",
        "lagetyp": "nicht beurteilbar",
        "intervalle": {"pq": "138 ms — normal", "qrs": "85 ms — normal",
                       "qt": "362 ms", "qtc": "419 ms — normal"},
        "befund": (
            "Normofrequenter Sinusrhythmus (80/min) mit unauffälligen Zeitintervallen: "
            "PQ-Zeit (138 ms), QRS-Komplex (85 ms) und QTc (419 ms) im Normbereich. Die "
            "KI-Analyse favorisiert nur knapp eine Hypertrophie/Belastung (HYP, 49 %), "
            "praktisch gleichauf mit einem Normalbefund (46 %). Aus den vorliegenden "
            "Parametern sind keine Hypertrophiezeichen ableitbar."
        ),
        "diagnosen": ["Hypertrophie/Belastung (fraglich)"],
        "beurteilung": (
            "Unsicherer Befund: Hypertrophie und Normalbefund sind nahezu gleich "
            "wahrscheinlich. Eine Hypertrophie ist nicht gesichert; Beurteilung der "
            "Kammerkomplex-Amplituden durch den Arzt erforderlich."
        ),
        "hinweis": (
            "Sehr niedrige Klassifikationskonfidenz (HYP 49 % vs. NORM 46 %) — dieser "
            "Befund ist nicht belastbar und muss ärztlich verifiziert werden."
        ),
    },
    175: {
        "rhythmus": "Sinusrhythmus, bradykard (55/min)",
        "lagetyp": "nicht beurteilbar",
        "intervalle": {"pq": "85 ms — unterer Grenzbereich", "qrs": "148 ms — verbreitert",
                       "qt": "480 ms", "qtc": "460 ms — verlängert"},
        "befund": (
            "Bradykarder Sinusrhythmus (55/min) mit verbreitertem QRS-Komplex (148 ms) und "
            "verlängerter frequenzkorrigierter QT-Zeit (QTc 460 ms); die PQ-Zeit (85 ms) "
            "liegt im unteren Grenzbereich. Die KI-Analyse weist mit grenzwertiger "
            "Wahrscheinlichkeit auf einen Myokardinfarkt hin (MI, 63 %), bei knapper "
            "Abgrenzung zum Normalbefund (48 %)."
        ),
        "diagnosen": ["Myokardinfarkt (akut oder abgelaufen)"],
        "beurteilung": (
            "Grenzwertiger Infarkt-Verdacht bei verbreitertem Kammerkomplex und verlängerter "
            "QTc; angesichts niedriger Konfidenz und breiter Differenzialdiagnose (z. B. "
            "Schenkelblock) ärztliche Korrelation mit Klinik und Vorbefunden erforderlich."
        ),
        "hinweis": (
            "Niedrige Klassifikationskonfidenz (MI 63 %); breiter Kammerkomplex und "
            "verlängerte QTc sind unspezifisch und im klinischen Kontext zu bewerten."
        ),
    },
    9: {
        "rhythmus": "Sinusrhythmus, normofrequent (61/min)",
        "lagetyp": "nicht beurteilbar",
        "intervalle": {"pq": "152 ms — normal", "qrs": "104 ms — normal",
                       "qt": "378 ms", "qtc": "381 ms — normal"},
        "befund": (
            "Normofrequenter Sinusrhythmus (61/min) mit regelrechten Zeitintervallen: "
            "PQ-Zeit (152 ms), QRS-Komplex (104 ms) und QTc (381 ms) im Normbereich. Keine "
            "Hinweise auf Erregungsleitungs- oder Repolarisationsstörungen."
        ),
        "diagnosen": ["Normalbefund"],
        "beurteilung": (
            "Unauffälliges Ruhe-EKG ohne pathologischen Befund. Keine akute "
            "Handlungsdringlichkeit."
        ),
        "hinweis": (
            "Die KI-Klassifikation stützt einen Normalbefund mit hoher Wahrscheinlichkeit "
            "(96 %); die endgültige Beurteilung erfolgt durch den befundenden Arzt."
        ),
    },
    128: {
        "rhythmus": "Sinusrhythmus, grenzwertig tachykard (94/min)",
        "lagetyp": "nicht beurteilbar",
        "intervalle": {"pq": "123 ms — normal", "qrs": "78 ms — normal",
                       "qt": "nicht verwertbar (Messartefakt)",
                       "qtc": "nicht verwertbar (Messartefakt)"},
        "befund": (
            "Grenzwertig tachykarder Sinusrhythmus (94/min) mit regelrechter PQ-Zeit "
            "(123 ms) und schmalem QRS-Komplex (78 ms). Die automatische QT- und "
            "QTc-Bestimmung lieferte physiologisch unmögliche Werte und wurde als "
            "Messartefakt verworfen; eine Beurteilung der Repolarisationsdauer ist aus "
            "dieser Aufzeichnung nicht möglich. Die KI-Analyse favorisiert nur knapp "
            "ST-Strecken-/T-Wellen-Veränderungen (STTC, 36 %), nahezu gleichauf mit einem "
            "Normalbefund (31 %)."
        ),
        "diagnosen": ["ST-Strecken-/T-Wellen-Veränderungen (fraglich)"],
        "beurteilung": (
            "Unsicherer Befund: STTC und Normalbefund nahezu gleich wahrscheinlich; zudem "
            "ist die QT-Strecke wegen eines Messartefakts nicht beurteilbar. Bei junger "
            "Patientin ärztliche Durchsicht des Original-EKGs empfohlen."
        ),
        "hinweis": (
            "Sehr niedrige Konfidenz (STTC 36 % vs. NORM 31 %) UND nicht verwertbare "
            "QT-Messung — Befund nur unter ausdrücklichem ärztlichem Vorbehalt."
        ),
    },
    271: {
        "rhythmus": "Sinusrhythmus, grenzwertig tachykard (94/min)",
        "lagetyp": "nicht beurteilbar",
        "intervalle": {"pq": "118 ms — unterer Grenzbereich", "qrs": "110 ms — normal",
                       "qt": "241 ms", "qtc": "301 ms"},
        "befund": (
            "Grenzwertig tachykarder Sinusrhythmus (94/min) mit noch normaler QRS-Dauer "
            "(110 ms); die PQ-Zeit (118 ms) liegt im unteren Grenzbereich. Die KI-Analyse "
            "klassifiziert ein kombiniertes Befundmuster: Myokardinfarkt (MI, 95 %), "
            "ST-Strecken-/T-Wellen-Veränderungen (STTC, 87 %) und Hypertrophie/Belastung "
            "(HYP, 57 %). Die niedrige QTc (301 ms) ist durch die höhere Herzfrequenz "
            "mitbestimmt."
        ),
        "diagnosen": [
            "Myokardinfarkt (akut oder abgelaufen)",
            "ST-Strecken-/T-Wellen-Veränderungen",
            "Hypertrophie/Belastung",
        ],
        "beurteilung": (
            "Hochauffälliges EKG mit gleichzeitigem Hinweis auf Infarkt, Repolarisations"
            "störung und Hypertrophie bei hoher Konfidenz für MI und STTC; zeitnahe "
            "ärztliche Korrelation mit Klinik, Vorbefunden und 12-Kanal-Morphologie "
            "erforderlich."
        ),
        "hinweis": (
            "Multi-Label-Befund mit drei gleichzeitigen Pathologien; HYP nur grenzwertig "
            "(57 %). Befundung und Therapieentscheidung ausschließlich durch den Arzt."
        ),
    },
}

# ─── Hauptprogramm ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"=== Lade Faktum-Sets aus {FACTSET_FILE} ===")
    with open(FACTSET_FILE, encoding="utf-8") as f:
        fact_sets = json.load(f)

    results = []
    for fs in fact_sets:
        ecg_id = fs["ecg_id"]
        print(f"\n=== EKG {ecg_id} | true: {fs['true_classes']} | pred: {fs['predicted_classes']} ===")
        befund = BEFUNDE.get(ecg_id)
        if befund is None:
            print(f"  (kein lokaler Befund für ecg_id {ecg_id} hinterlegt — übersprungen)")
            continue

        raw_output = json.dumps(befund, ensure_ascii=False, indent=2)
        issues = consistency_check(fs, raw_output)
        plaus  = check_features(fs["classical_features"])

        results.append({
            "ecg_id": ecg_id,
            "true_classes": fs["true_classes"],
            "predicted_classes": fs["predicted_classes"],
            "befund": befund,
            "konsistenz": {"ok": len(issues) == 0, "issues": issues},
            "plausibilitaet": {"alle_verwertbar": plaus["alle_verwertbar"],
                               "artefakte": plaus["artefakte"]},
            "backend": BACKEND,
        })

        print(json.dumps(befund, indent=2, ensure_ascii=False))
        print("✓ Konsistenz ok" if not issues else f"⚠ Konsistenz-Issues: {issues}")
        if not plaus["alle_verwertbar"]:
            print(f"⚠ Plausibilität: {plaus['n_artefakte']} Artefakt(e) im Faktum-Set "
                  f"(im Befund korrekt als nicht verwertbar markiert)")

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for r in results if r["konsistenz"]["ok"])
    print(f"\n=== {len(results)} Befunde gespeichert: {OUTFILE} ===")
    print(f"=== Konsistenzcheck bestanden: {n_ok}/{len(results)} ===")
