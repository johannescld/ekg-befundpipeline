"""
Auswertung des Experiments "Pipeline gegen direkte Bildbefundung".

Was hier gemessen wird — und was nicht
--------------------------------------
Die beiden Arme sind nicht in jeder Hinsicht symmetrisch vergleichbar, und
das muss die Auswertung offenlegen statt verwischen:

  Messwerte      Arm A bekommt sie vorgegeben und kann sie nur korrekt oder
                 fehlerhaft übernehmen. Arm B muss sie vom Bild ablesen.
                 Ein direkter Vergleich der Abweichungen benachteiligt Arm B
                 strukturell — er beantwortet die Frage "wie genau lässt sich
                 ein EKG-Bild ablesen", nicht "welcher Arm ist besser".

  Halluzination  Fair für beide: Nennt der Text eine Zahl für eine Grösse,
                 die nicht sicher bestimmbar ist? Arm A dürfte das per
                 Konstruktion nicht (Prüfung in Stufe 3), Arm B hat keine
                 solche Sicherung. Das ist der eigentliche Prüfpunkt.

  Diagnose       Fair für beide: Übereinstimmung mit der unabhängigen
                 Kardiologen-Annotation aus PTB-XL.

  Konsistenz     Fair für beide: Widerspricht sich der Text selbst
                 (etwa "Zeiten normwertig" bei zugleich genannter
                 QRS-Dauer von 150 ms)?

Referenz für die Messwerte ist die in validate_morphology.py geprüfte
Signalanalyse. Sie ist nicht fehlerfrei — ihre Grenzen (insbesondere die
QRS-Dauer bei Linksschenkelblock) sind dokumentiert und gehen in die
Interpretation ein.

Aufruf:  python3 evaluate_faktentreue.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ekg_config import RESULTS

EXP_DIR = RESULTS / "experiment_bild_vs_pipeline"

# Zahlen aus dem Befundtext ziehen. Abstand zum Parameternamen begrenzt,
# damit nicht eine weit entfernte Zahl zugeordnet wird.
MUSTER = {
    "hr_bpm":          r"(\d{2,3})\s*(?:/min|bpm)",
    "pr_interval_ms":  r"\bPQ[^0-9]{0,20}?(\d{2,3})\s*ms",
    "qrs_duration_ms": r"\bQRS[^0-9]{0,20}?(\d{2,3})\s*ms",
    "qtc_bazett_ms":   r"\bQTc[^0-9]{0,20}?(\d{3})\s*ms",
}

NICHT_BESTIMMBAR = re.compile(
    r"nicht (?:sicher )?(?:beurteilbar|bestimmbar|verwertbar)|"
    r"nicht möglich|Messartefakt", re.IGNORECASE)

LAGETYPEN = ["überdrehter Linkstyp", "Linkstyp", "Indifferenztyp",
             "Steiltyp", "überdrehter Rechtstyp", "Rechtstyp"]


def text_aus(befund) -> str:
    """Fügt alle Textfelder eines Befund-JSON zusammen."""
    if befund is None:
        return ""
    if isinstance(befund, str):
        return befund
    teile = []
    for v in befund.values():
        if isinstance(v, str):
            teile.append(v)
        elif isinstance(v, dict):
            teile += [x for x in v.values() if isinstance(x, str)]
    return " ".join(teile)


def lagetyp_aus(text: str):
    for lt in LAGETYPEN:          # längere Begriffe zuerst prüfen
        if lt.lower() in text.lower():
            return lt
    return None


def bewerte(text: str, fs: dict) -> dict:
    """Vergleicht einen Befundtext mit dem geprüften Faktum-Set."""
    feat = fs.get("classical_features_verwertbar") or fs["classical_features"]
    morph = fs.get("morphologie", {})
    out = {}

    # ── Messwerte: Abweichung und Halluzination ─────────────────────────────
    halluziniert, abweichungen = [], {}
    for param, muster in MUSTER.items():
        treffer = re.findall(muster, text, re.IGNORECASE)
        genannt = float(treffer[0]) if treffer else None
        soll = feat.get(param)

        if genannt is not None and soll is None:
            # Wert genannt, obwohl er nicht sicher bestimmbar ist
            halluziniert.append(param)
        elif genannt is not None and soll is not None:
            abweichungen[param] = abs(genannt - float(soll))
    out["halluzinierte_werte"] = halluziniert
    out["n_halluziniert"] = len(halluziniert)
    out["abweichungen"] = abweichungen

    # ── Lagetyp ─────────────────────────────────────────────────────────────
    soll_lt = (morph.get("lagetyp") or {}).get("lagetyp")
    ist_lt = lagetyp_aus(text)
    out["lagetyp_genannt"] = ist_lt
    out["lagetyp_soll"] = soll_lt
    out["lagetyp_korrekt"] = (None if not (soll_lt and ist_lt)
                              else ist_lt == soll_lt)

    # ── Pardée-Q ────────────────────────────────────────────────────────────
    q = morph.get("q_zacken", {})
    soll_q = bool(q.get("pardee_q_vorhanden"))
    verneint = re.search(r"kein[e]?\s+(?:pathologischen?\s+)?"
                         r"(?:Pardée-?Q|Q-Zacken)", text, re.IGNORECASE)
    bejaht = re.search(r"pathologische[rn]?\s+Q-Zacken|Pardée-?Q\s+in",
                       text, re.IGNORECASE)
    ist_q = True if bejaht else (False if verneint else None)
    out["pardee_q_korrekt"] = None if ist_q is None else (ist_q == soll_q)

    # ── Interne Konsistenz ──────────────────────────────────────────────────
    probleme = []
    m_qrs = re.search(MUSTER["qrs_duration_ms"], text, re.IGNORECASE)
    if m_qrs and float(m_qrs.group(1)) >= 120 and \
            re.search(r"QRS[^.]{0,40}normwertig", text, re.IGNORECASE):
        probleme.append("QRS >= 120 ms zugleich als normwertig bezeichnet")
    if re.search(r"\bkeine? .{0,25}Extrasystole", text, re.IGNORECASE) and \
            re.search(r"\d+\s+Extrasystole", text, re.IGNORECASE):
        probleme.append("Extrasystolen zugleich verneint und gezählt")
    out["konsistenzprobleme"] = probleme

    # ── Modellsprache (darf in keinem Befund stehen) ────────────────────────
    modellsprache = re.findall(
        r"\d{1,3}\s*%|\bKI\b|\bModell\b|\bKlassifik\w*|\bWahrscheinlichkeit von\b",
        text, re.IGNORECASE)
    out["modellsprache"] = modellsprache

    # ── Diagnose gegen PTB-XL ───────────────────────────────────────────────
    from label_mapping_hybrid import TARGET_LABELS_DE, TARGET_TO_SUPER
    wahr = set(fs.get("true_classes", []))
    wahr_super = {TARGET_TO_SUPER[c] for c in wahr if c in TARGET_TO_SUPER}
    stichworte = {
        "NORM": ["normalbefund", "unauffällig"],
        "MI":   ["infarkt"],
        "STTC": ["ischämie", "st-t", "st-strecken", "repolarisation"],
        "CD":   ["schenkelblock", "hemiblock", "av-block",
                 "leitungsverzögerung", "erregungsleitung"],
        "HYP":  ["hypertrophie", "belastung"],
    }
    erkannt = {s for s, w in stichworte.items()
               if any(x in text.lower() for x in w)}
    out["diagnose_super_soll"] = sorted(wahr_super)
    out["diagnose_super_genannt"] = sorted(erkannt)
    out["diagnose_treffer"] = len(wahr_super & erkannt)
    out["diagnose_verfehlt"] = len(wahr_super - erkannt)
    out["diagnose_zusatz"] = len(erkannt - wahr_super)
    return out


def selbsttest():
    """
    Prüft die Auswertungslogik an konstruierten Texten mit bekannten Fehlern.

    Kein Modelllauf und keine Aussage über Befundqualität — es wird nur
    geprüft, ob die Extraktion findet, was sie finden soll. Ohne diesen Test
    wäre unklar, ob eine Halluzinationsrate von 0 echt ist oder nur bedeutet,
    dass der reguläre Ausdruck nichts erkennt.
    """
    import json as _json
    with open(EXP_DIR / "fact_sets.json", encoding="utf-8") as f:
        fact_sets = {fs["ecg_id"]: fs for fs in _json.load(f)}
    fs = next(iter(fact_sets.values()))
    feat = fs.get("classical_features_verwertbar") or fs["classical_features"]

    faelle = [
        ("korrekte Übernahme",
         f"Normokarder Sinusrhythmus, Frequenz {feat['hr_bpm']:.0f}/min. "
         f"PQ-Zeit {feat['pr_interval_ms']:.0f} ms normwertig. "
         f"Kein Pardée-Q. Keine Extrasystolen.",
         {"n_halluziniert": 0, "modellsprache": 0}),
        ("erfundener Wert",
         "Frequenz 70/min. QTc-Zeit 410 ms normwertig. Kein Pardée-Q.",
         {"erwartung": "QTc nur dann als Halluzination, wenn im Faktum-Set "
                       "nicht bestimmbar"}),
        ("Modellsprache",
         "Die KI klassifiziert mit 97 % einen Vorderwandinfarkt. Das Modell "
         "sagt Frequenz 80/min.",
         {"modellsprache_min": 3}),
        ("Selbstwiderspruch",
         "QRS-Zeit 150 ms, QRS normwertig. Keine Extrasystolen, "
         "3 Extrasystolen nachweisbar.",
         {"konsistenz_min": 2}),
        ("Diagnose erkannt",
         "Beurteilung: Nachweis eines Vorderwandinfarktes und einer "
         "linksventrikulären Hypertrophie.",
         {}),
    ]

    print("=" * 78)
    print(f"  Selbsttest der Auswertungslogik (Faktum-Set EKG {fs['ecg_id']})")
    print("=" * 78)
    print(f"Referenzwerte: HR={feat.get('hr_bpm')} PQ={feat.get('pr_interval_ms')} "
          f"QRS={feat.get('qrs_duration_ms')} QTc={feat.get('qtc_bazett_ms')}")
    print(f"Pardée-Q laut Messung: "
          f"{fs['morphologie']['q_zacken'].get('pardee_q_vorhanden')}\n")

    for name, text, _ in faelle:
        b = bewerte(text, fs)
        print(f"{name}")
        print(f"   halluziniert: {b['n_halluziniert']} {b['halluzinierte_werte']}")
        print(f"   Abweichungen: {({k: round(v,1) for k,v in b['abweichungen'].items()})}")
        print(f"   Pardée-Q korrekt: {b['pardee_q_korrekt']}   "
              f"Lagetyp: {b['lagetyp_genannt']}")
        print(f"   Modellsprache: {len(b['modellsprache'])} {b['modellsprache'][:3]}")
        print(f"   Konsistenzprobleme: {b['konsistenzprobleme']}")
        print(f"   Diagnose: getroffen {b['diagnose_treffer']}, "
              f"verfehlt {b['diagnose_verfehlt']}, "
              f"zusätzlich {b['diagnose_zusatz']} "
              f"(soll {b['diagnose_super_soll']})")
        print()


def main():
    import sys
    if "--selbsttest" in sys.argv:
        selbsttest()
        return
    erg_datei = EXP_DIR / "ergebnisse.json"
    if not erg_datei.exists():
        print(f"Keine Ergebnisse unter {erg_datei}.")
        print("Zuerst den Modelllauf ausführen:")
        print("  ANTHROPIC_API_KEY=sk-ant-... python3 experiment_pipeline_vs_bild.py")
        return

    with open(erg_datei, encoding="utf-8") as f:
        ergebnisse = json.load(f)
    with open(EXP_DIR / "fact_sets.json", encoding="utf-8") as f:
        fact_sets = {fs["ecg_id"]: fs for fs in json.load(f)}

    zeilen = []
    for e in ergebnisse:
        fs = fact_sets.get(e["ecg_id"])
        if fs is None:
            continue
        for arm, schluessel in [("Pipeline", "arm_a_pipeline"),
                                ("Bild", "arm_b_bild")]:
            b = bewerte(text_aus(e[schluessel].get("befund")), fs)
            zeilen.append({"ecg_id": e["ecg_id"], "arm": arm, **b,
                           "tokens_out": e[schluessel].get("tokens_out")})

    df = pd.DataFrame(zeilen)
    df.to_json(EXP_DIR / "bewertung_roh.json", orient="records", indent=2,
               force_ascii=False)

    print("=" * 78)
    print(f"  Faktentreue — {df.ecg_id.nunique()} EKGs, zwei Arme")
    print("=" * 78)

    tab = []
    for arm in ["Pipeline", "Bild"]:
        d = df[df.arm == arm]
        abw = {}
        for p in MUSTER:
            werte = [r.get(p) for r in d.abweichungen if p in r]
            abw[p] = float(np.mean(werte)) if werte else np.nan
        tab.append({
            "Arm": arm,
            "halluz. Werte je Befund": d.n_halluziniert.mean(),
            "Befunde mit Halluzination": (d.n_halluziniert > 0).mean(),
            "Lagetyp korrekt": d.lagetyp_korrekt.dropna().mean()
                if d.lagetyp_korrekt.notna().any() else np.nan,
            "Pardée-Q korrekt": d.pardee_q_korrekt.dropna().mean()
                if d.pardee_q_korrekt.notna().any() else np.nan,
            "Diagnosen getroffen": d.diagnose_treffer.mean(),
            "Diagnosen verfehlt": d.diagnose_verfehlt.mean(),
            "Zusatzdiagnosen": d.diagnose_zusatz.mean(),
            "Konsistenzprobleme": d.konsistenzprobleme.map(len).mean(),
            "Modellsprache": d.modellsprache.map(len).mean(),
            **{f"Δ {p}": abw[p] for p in MUSTER},
        })
    zus = pd.DataFrame(tab).set_index("Arm")
    pd.set_option("display.width", 200)
    print(zus.T.round(3).to_string())
    zus.to_csv(EXP_DIR / "faktentreue_zusammenfassung.csv")

    print("\nLesart:")
    print("  Halluzination = Zahl genannt, obwohl die Grösse laut Prüfung nicht")
    print("  sicher bestimmbar ist. Das ist der fairste Vergleichspunkt.")
    print("  Die Δ-Werte messen die Abweichung zur Signalanalyse; für den")
    print("  Pipeline-Arm ist das die Treue zur Vorgabe, für den Bild-Arm die")
    print("  Genauigkeit des Ablesens — beides sind verschiedene Fragen.")

    # ── Abbildung ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    farben = {"Pipeline": "#4C72B0", "Bild": "#C44E52"}

    ax = axes[0]
    w = [df[df.arm == a].n_halluziniert.mean() for a in ["Pipeline", "Bild"]]
    ax.bar(["Pipeline", "Bild"], w, color=[farben[a] for a in ["Pipeline", "Bild"]])
    ax.set_ylabel("erfundene Werte je Befund")
    ax.set_title("A  Halluzinierte Messwerte")
    ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    breite, x = 0.35, np.arange(len(MUSTER))
    for i, a in enumerate(["Pipeline", "Bild"]):
        d = df[df.arm == a]
        werte = [np.mean([r.get(p) for r in d.abweichungen if p in r])
                 if any(p in r for r in d.abweichungen) else 0 for p in MUSTER]
        ax.bar(x + i * breite, werte, breite, label=a, color=farben[a])
    ax.set_xticks(x + breite / 2)
    ax.set_xticklabels(["HR", "PQ", "QRS", "QTc"])
    ax.set_ylabel("mittlere Abweichung")
    ax.set_title("B  Abweichung zur Signalanalyse")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)

    ax = axes[2]
    for i, a in enumerate(["Pipeline", "Bild"]):
        d = df[df.arm == a]
        ax.bar([i - 0.2, i + 0.2],
               [d.diagnose_treffer.mean(), d.diagnose_verfehlt.mean()],
               0.35, color=[farben[a], "#BBBBBB"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pipeline", "Bild"])
    ax.set_ylabel("Diagnosen je Befund")
    ax.set_title("C  getroffen (farbig) / verfehlt (grau)")
    ax.grid(axis="y", alpha=.3)

    fig.suptitle("Strukturierte Pipeline gegen direkte Bildbefundung",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "faktentreue.png", dpi=180, bbox_inches="tight")
    fig.savefig(EXP_DIR / "faktentreue.pdf", bbox_inches="tight")
    print(f"\nGespeichert: {EXP_DIR / 'faktentreue_zusammenfassung.csv'}")
    print(f"             {EXP_DIR / 'faktentreue.pdf'}")


if __name__ == "__main__":
    main()
