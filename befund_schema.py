"""
Befundschema Version 2 — strukturierter Aufbau nach kardiologischer
Lehrbuchkonvention (Vorgabe des Betreuers, 08/2026).

Zwei Änderungen gegenüber Version 1:

1. AUFBAU. Der Befund folgt der klinischen Reihenfolge
   Rhythmus → Frequenz → Lagetyp → Zeiten → Q-Zacken → R-Progression
   → ST-Strecke → T-Wellen → Extrasystolen → Beurteilung.
   Vorbild ist die übliche Formulierung eines Normalbefunds:

     "Normokarder Sinusrhythmus, durchschnittliche Frequenz 68/min,
      Indifferenztyp. PQ-, QRS- sowie QTc-Zeiten normwertig. Kein
      Pardée-Q. Normale R-Progredienz, keine S-Persistenz in V6,
      R/S-Umschlag zwischen V3 und V4, keine relevanten ST-Hebungen
      oder Senkungen, keine T-Negativierungen, keine Extrasystolen."

2. KEIN MODELL-SPRECH. Klassifikationswahrscheinlichkeiten erscheinen
   nicht mehr im Befundtext. Statt "Das Modell hat X mit 79 % vorhergesagt"
   steuert die Konfidenz die sprachliche Sicherheitsstufe:

     >= 0.85   "Nachweis eines ..."        (SICHER)
     >= 0.65   "Wahrscheinlich ..."        (WAHRSCHEINLICH)
     >= 0.50   "Vereinbar mit ..."         (MOEGLICH)
     <  0.50   erscheint nicht im Befund

   Die numerischen Werte bleiben im Protokoll (`qualitaetssicherung`)
   erhalten — für die Auswertung, nicht für den Befundleser.
"""

from __future__ import annotations   # erlaubt 'str | None' unter Python 3.9

from label_mapping_hybrid import TARGET_LABELS_DE, TARGET_TO_SUPER

# ─── Sicherheitsstufen ───────────────────────────────────────────────────────
SICHER, WAHRSCHEINLICH, MOEGLICH = "sicher", "wahrscheinlich", "möglich"

SCHWELLEN = [(0.85, SICHER), (0.65, WAHRSCHEINLICH), (0.50, MOEGLICH)]

FORMULIERUNG = {
    SICHER:         "Nachweis {gen}",
    WAHRSCHEINLICH: "Wahrscheinlich {nom}",
    MOEGLICH:       "Vereinbar mit {dat}",
}

# Superklassen-Klartext für den Fallback bei unsicherer Subklasse
SUPER_LABELS_DE = {
    "NORM": "Normalbefund",
    "MI":   "Myokardinfarkt",
    "STTC": "ST-T-Streckenveränderung",
    "CD":   "Erregungsleitungsstörung",
    "HYP":  "Hypertrophie oder Belastung",
}

# Befundtaugliche Bezeichnung je Zielklasse, dekliniert
# (Genitiv für "Nachweis ...", Nominativ für "Wahrscheinlich ...",
#  Dativ für "Vereinbar mit ...").
# Die Bezeichnungen aus TARGET_LABELS_DE sind für Fließtext ungeeignet
# ("unspezifische ST-T-Veränderungen" im Plural, "Ischämie, unspezifisch"
# mit nachgestelltem Attribut) und werden hier ersetzt.
KASUS_KLASSE = {
    "NORM":      ("eines Normalbefundes", "ein Normalbefund",
                  "einem Normalbefund"),
    "AMI":       ("eines Vorderwandinfarktes", "ein Vorderwandinfarkt",
                  "einem Vorderwandinfarkt"),
    "IMI":       ("eines Hinterwandinfarktes", "ein Hinterwandinfarkt",
                  "einem Hinterwandinfarkt"),
    "MI_other":  ("eines lateralen oder posterioren Myokardinfarktes",
                  "ein lateraler oder posteriorer Myokardinfarkt",
                  "einem lateralen oder posterioren Myokardinfarkt"),
    "STTC":      ("unspezifischer ST-T-Streckenveränderungen",
                  "unspezifische ST-T-Streckenveränderungen",
                  "unspezifischen ST-T-Streckenveränderungen"),
    "NST_":      ("unspezifischer ST-Streckenveränderungen",
                  "unspezifische ST-Streckenveränderungen",
                  "unspezifischen ST-Streckenveränderungen"),
    "ISC_":      ("einer Ischämie", "eine Ischämie", "einer Ischämie"),
    "ISCA":      ("einer Ischämie der Vorderwand",
                  "eine Ischämie der Vorderwand",
                  "einer Ischämie der Vorderwand"),
    "ISCI":      ("einer Ischämie der Hinterwand",
                  "eine Ischämie der Hinterwand",
                  "einer Ischämie der Hinterwand"),
    "LAFB/LPFB": ("eines Hemiblocks", "ein Hemiblock", "einem Hemiblock"),
    "IRBBB":     ("eines inkompletten Rechtsschenkelblocks",
                  "ein inkompletter Rechtsschenkelblock",
                  "einem inkompletten Rechtsschenkelblock"),
    "CRBBB":     ("eines kompletten Rechtsschenkelblocks",
                  "ein kompletter Rechtsschenkelblock",
                  "einem kompletten Rechtsschenkelblock"),
    "CLBBB":     ("eines kompletten Linksschenkelblocks",
                  "ein kompletter Linksschenkelblock",
                  "einem kompletten Linksschenkelblock"),
    "_AVB":      ("eines AV-Blocks", "ein AV-Block", "einem AV-Block"),
    "IVCD":      ("einer intraventrikulären Leitungsverzögerung",
                  "eine intraventrikuläre Leitungsverzögerung",
                  "einer intraventrikulären Leitungsverzögerung"),
    "CD_other":  ("einer sonstigen Erregungsleitungsstörung",
                  "eine sonstige Erregungsleitungsstörung",
                  "einer sonstigen Erregungsleitungsstörung"),
    "LVH":       ("einer linksventrikulären Hypertrophie",
                  "eine linksventrikuläre Hypertrophie",
                  "einer linksventrikulären Hypertrophie"),
    "LAO/LAE":   ("einer linksatrialen Belastung",
                  "eine linksatriale Belastung",
                  "einer linksatrialen Belastung"),
    "HYP_other": ("einer rechtsseitigen oder septalen Hypertrophie",
                  "eine rechtsseitige oder septale Hypertrophie",
                  "einer rechtsseitigen oder septalen Hypertrophie"),
    # Superklassen (Ziel des Konfidenz-Gatings)
    "MI":        ("eines Myokardinfarktes", "ein Myokardinfarkt",
                  "einem Myokardinfarkt"),
    "CD":        ("einer Erregungsleitungsstörung",
                  "eine Erregungsleitungsstörung",
                  "einer Erregungsleitungsstörung"),
    "HYP":       ("einer Hypertrophie oder Belastung",
                  "eine Hypertrophie oder Belastung",
                  "einer Hypertrophie oder Belastung"),
}


def sicherheitsstufe(p: float) -> str | None:
    """Ordnet einer Wahrscheinlichkeit die sprachliche Sicherheitsstufe zu."""
    for schwelle, stufe in SCHWELLEN:
        if p >= schwelle:
            return stufe
    return None


def formuliere_diagnose(klasse: str, p: float) -> str | None:
    """
    Wandelt Klassenschlüssel + Wahrscheinlichkeit in eine Befundformulierung.
    Gibt None zurück, wenn die Klasse nicht in den Befund gehört.
    """
    stufe = sicherheitsstufe(p)
    if stufe is None:
        return None
    bez = TARGET_LABELS_DE.get(klasse, SUPER_LABELS_DE.get(klasse, klasse))
    gen, nom, dat = KASUS_KLASSE.get(
        klasse, (f"einer {bez}", f"eine {bez}", f"einer {bez}"))
    return FORMULIERUNG[stufe].format(gen=gen, nom=nom, dat=dat)


def frequenz_attribut(hr: float | None) -> str:
    """Normokard / brady- / tachykard."""
    if hr is None:
        return "Frequenz nicht bestimmbar"
    if hr < 60:
        return "bradykard"
    if hr > 100:
        return "tachykard"
    return "normokard"


# ─── Befundschema (JSON-Struktur, die das Sprachmodell füllen soll) ──────────
BEFUND_SCHEMA = {
    "rhythmus": "<Rhythmusart mit Frequenzattribut, z. B. 'Normokarder Sinusrhythmus'>",
    "frequenz_bpm": "<Zahl>",
    "lagetyp": "<Indifferenztyp/Linkstyp/Steiltyp/... oder 'nicht bestimmbar'>",
    "zeiten": {
        "pq": "<Wert ms — normwertig/verlängert/verkürzt oder 'nicht bestimmbar'>",
        "qrs": "<Wert ms — normwertig/verbreitert>",
        "qtc": "<Wert ms — normwertig/verlängert>",
    },
    "q_zacken": "<'Kein Pardée-Q.' oder Beschreibung mit Ableitungsgruppe>",
    "r_progression": "<R-Progredienz, R/S-Umschlag, S-Persistenz V6>",
    "st_strecke": "<'Keine relevanten ST-Hebungen oder -Senkungen.' oder Beschreibung>",
    "t_wellen": "<'Keine T-Negativierungen.' oder Beschreibung>",
    "extrasystolen": "<'Keine Extrasystolen.' oder Anzahl und Typ>",
    "befundtext": "<Fließtext, der die obigen Punkte in der genannten "
                  "Reihenfolge zusammenfasst — Lehrbuchstil>",
    "beurteilung": "<1-2 Sätze. Beginnt mit der Sicherheitsstufe, z. B. "
                   "'Wahrscheinlich Vorderwandinfarkt.' Keine Prozentangaben.>",
    "limitation": "<optional: was nicht beurteilbar war und warum>",
}


SYSTEM_PROMPT = """Du bist ein erfahrener Kardiologe und formulierst EKG-Befunde \
für die klinische Dokumentation.

Du erhältst ein Faktum-Set mit Messwerten aus einer automatisierten \
Signalanalyse sowie eine bereits sprachlich eingestufte Verdachtsdiagnose.

AUFBAU DES BEFUNDES — halte diese Reihenfolge strikt ein:
  1. Rhythmus und Frequenz
  2. Lagetyp
  3. Zeiten (PQ, QRS, QTc)
  4. Q-Zacken (Pardée-Q)
  5. R-Progredienz, R/S-Umschlag, S-Persistenz in V6
  6. ST-Strecke
  7. T-Wellen
  8. Extrasystolen
  9. Beurteilung

STIL — so klingt ein echter Befund:
  "Normokarder Sinusrhythmus, durchschnittliche Frequenz 68/min, \
Indifferenztyp. PQ-, QRS- sowie QTc-Zeiten normwertig. Kein Pardée-Q. \
Normale R-Progredienz, keine S-Persistenz in V6, R/S-Umschlag zwischen \
V3 und V4, keine relevanten ST-Hebungen oder Senkungen, keine \
T-Negativierungen, keine Extrasystolen.
  Beurteilung: Unauffälliges Ruhe-EKG."

ABSOLUTE REGELN:
- Schreibe NIEMALS über das Modell, die KI, eine Klassifikation oder eine \
Wahrscheinlichkeit. Keine Prozentangaben, keine Formulierungen wie \
"das Modell sagt" oder "mit 79 % Wahrscheinlichkeit". Der Befund liest sich \
wie von einem Arzt geschrieben.
- Die Beurteilung übernimmt WÖRTLICH die im Faktum-Set vorgegebene \
Formulierung der Verdachtsdiagnose ("Nachweis eines ...", \
"Wahrscheinlich ...", "Vereinbar mit ..."). Verschärfe und relativiere sie nicht.
- Verwende AUSSCHLIESSLICH Werte aus dem Faktum-Set. Erfinde nichts.
- Was im Faktum-Set als "nicht bestimmbar" markiert ist, wird als nicht \
beurteilbar benannt — niemals als unauffällig.
- Negativbefunde gehören ausdrücklich in den Text ("Kein Pardée-Q.", \
"Keine Extrasystolen."), wie im klinischen Befund üblich.
- Sprache: medizinisches Deutsch, knapp, ohne Füllwörter.
- Antworte NUR mit dem JSON-Objekt."""


def _fmt(wert, einheit="", nachkomma=0):
    if wert is None:
        return "nicht bestimmbar"
    return f"{wert:.{nachkomma}f}{einheit}"


def _zeit(wert, unter, ober, name_unter="verkürzt", name_ober="verlängert"):
    """Formatiert einen Zeitwert samt Bewertung, oder meldet ihn als fehlend."""
    if wert is None:
        return "nicht bestimmbar (Messartefakt oder nicht auswertbar)"
    return f"{wert:.0f} ms — {_bewertung(wert, unter, ober, name_unter, name_ober)}"


def _bewertung(wert, unter, ober, name_unter="verkürzt", name_ober="verlängert"):
    if wert is None:
        return "nicht bestimmbar"
    if unter is not None and wert < unter:
        return name_unter
    if ober is not None and wert > ober:
        return name_ober
    return "normwertig"


def build_user_prompt(fs: dict) -> str:
    """Baut den User-Prompt aus einem erweiterten Faktum-Set (Version 2)."""
    pat = fs.get("patient", {})
    # Plausibilitätsgeprüfte Werte verwenden: als Messartefakt verworfene
    # Parameter stehen dort auf None und erscheinen im Prompt als "nicht
    # bestimmbar" — sie dürfen nicht als Zahl in den Befund gelangen.
    feat = fs.get("classical_features_verwertbar") or fs.get("classical_features", {})
    morph = fs.get("morphologie", {})
    diag = fs.get("diagnose", {})

    hr = feat.get("hr_bpm")
    pq, qrs = feat.get("pr_interval_ms"), feat.get("qrs_duration_ms")
    qtc = feat.get("qtc_bazett_ms")

    rhythm = morph.get("rhythmus", {})
    lage = morph.get("lagetyp", {})
    q = morph.get("q_zacken", {})
    rp = morph.get("r_progression", {})
    st = morph.get("st_strecke", {})
    tw = morph.get("t_wellen", {})

    # ── Q-Zacken ────────────────────────────────────────────────────────────
    if not q.get("beurteilbar"):
        q_txt = "nicht beurteilbar"
    elif q.get("pardee_q_vorhanden"):
        q_txt = ("pathologische Q-Zacken in den Ableitungsgruppen: "
                 + ", ".join(q.get("betroffene_gruppen", []))
                 + f" (Ableitungen: {', '.join(q.get('pathologische_q_ableitungen', []))})")
    else:
        q_txt = "kein Pardée-Q"

    # ── R-Progression ───────────────────────────────────────────────────────
    if not rp.get("beurteilbar"):
        r_txt = "nicht beurteilbar"
    else:
        teile = []
        pn = rp.get("r_progression_normal")
        teile.append("R-Progredienz nicht bestimmbar" if pn is None else
                     "R-Progredienz regelrecht" if pn else
                     "verzögerte R-Progredienz")
        um = rp.get("rs_umschlag")
        teile.append(f"R/S-Umschlag in {um}" if um else "R/S-Umschlag nicht bestimmbar")
        teile.append("S-Persistenz in V6" if rp.get("s_persistenz_v6")
                     else "keine S-Persistenz in V6")
        r_txt = ", ".join(teile)

    # ── ST-Strecke ──────────────────────────────────────────────────────────
    if not st.get("beurteilbar"):
        st_txt = "nicht beurteilbar"
    else:
        teile = []
        if st.get("relevante_hebung"):
            teile.append(f"ST-Hebungen in {', '.join(st.get('hebungen', []))} "
                         f"(Gruppen: {', '.join(st.get('hebung_gruppen', []))})")
        if st.get("relevante_senkung"):
            teile.append(f"ST-Senkungen in {', '.join(st.get('senkungen', []))} "
                         f"(Gruppen: {', '.join(st.get('senkung_gruppen', []))})")
        st_txt = "; ".join(teile) if teile else \
            "keine relevanten ST-Hebungen oder -Senkungen"

    # ── T-Wellen ────────────────────────────────────────────────────────────
    if not tw.get("beurteilbar"):
        t_txt = "nicht beurteilbar"
    elif tw.get("relevante_negativierung"):
        t_txt = f"T-Negativierungen in {', '.join(tw.get('pathologisch_negativ', []))}"
    else:
        t_txt = "keine T-Negativierungen"

    # ── Extrasystolen ───────────────────────────────────────────────────────
    n_es = rhythm.get("extrasystolen_anzahl")
    if n_es is None:
        es_txt = "nicht beurteilbar"
    elif n_es == 0:
        es_txt = "keine Extrasystolen"
    else:
        es_txt = f"{n_es} vorzeitige Aktion(en)"
        if rhythm.get("extrasystolen_typ"):
            es_txt += f", {rhythm['extrasystolen_typ']}"

    # ── Verdachtsdiagnose (bereits sprachlich eingestuft) ────────────────────
    diag_zeilen = []
    for d in diag.get("formulierungen", []):
        diag_zeilen.append(f"  - {d}")
    if diag.get("vorbehalt"):
        diag_zeilen.append(f"  Vorbehalt (in die Limitation aufnehmen): "
                           f"{diag['vorbehalt']}")
    diag_txt = "\n".join(diag_zeilen) if diag_zeilen else \
        "  - keine Diagnose oberhalb der Befundschwelle"

    return f"""=== EKG-Faktum-Set (ecg_id: {fs.get('ecg_id')}) ===

Patient: {pat.get('age_years')} Jahre, {pat.get('sex')}
Aufzeichnung: {fs.get('acquisition', {}).get('duration_s')} s, \
{fs.get('acquisition', {}).get('sampling_rate_hz')} Hz, 12 Ableitungen

RHYTHMUS
  Frequenz: {_fmt(hr, '/min')} ({frequenz_attribut(hr)})
  Regelmäßigkeit: {'regelmäßig' if rhythm.get('regelmaessig') else 'unregelmäßig'}
  Extrasystolen: {es_txt}

LAGETYP
  {lage.get('lagetyp', 'nicht bestimmbar')} \
({_fmt(lage.get('achse_grad'), '°')})

ZEITEN
  PQ:  {_zeit(pq, 120, 200)}
  QRS: {_zeit(qrs, None, 120, name_ober='verbreitert')}
  QTc: {_zeit(qtc, 350, 440)}

MORPHOLOGIE
  Q-Zacken:        {q_txt}
  R-Progression:   {r_txt}
  ST-Strecke:      {st_txt}
  T-Wellen:        {t_txt}

VERDACHTSDIAGNOSE (bereits sprachlich eingestuft — wörtlich übernehmen)
{diag_txt}

Erstelle den strukturierten EKG-Befund als JSON nach dem vorgegebenen Schema."""


def render_befund(befund: dict) -> str:
    """Rendert das JSON-Befundobjekt als lesbaren Befundtext."""
    zeilen = []
    if befund.get("befundtext"):
        zeilen.append(befund["befundtext"].strip())
    else:  # Fallback: aus den Einzelfeldern zusammensetzen
        z = befund.get("zeiten", {})
        teile = [
            f"{befund.get('rhythmus', '')}, Frequenz {befund.get('frequenz_bpm', '?')}/min, "
            f"{befund.get('lagetyp', '')}.",
            f"PQ {z.get('pq', '?')}, QRS {z.get('qrs', '?')}, QTc {z.get('qtc', '?')}.",
            befund.get("q_zacken", ""), befund.get("r_progression", ""),
            befund.get("st_strecke", ""), befund.get("t_wellen", ""),
            befund.get("extrasystolen", ""),
        ]
        zeilen.append(" ".join(t for t in teile if t))

    if befund.get("beurteilung"):
        zeilen.append(f"\nBeurteilung: {befund['beurteilung'].strip()}")
    if befund.get("limitation"):
        zeilen.append(f"\nEinschränkung: {befund['limitation'].strip()}")
    return "\n".join(zeilen)


def _gross(s: str) -> str:
    """Erster Buchstabe gross, Rest unveraendert.

    str.capitalize() waere falsch: es schreibt den Rest klein und macht aus
    "R/S-Umschlag in V3" ein "r/s-umschlag in v3".
    """
    return s[:1].upper() + s[1:] if s else s


# ─── Regelbasierter Referenzbefund ───────────────────────────────────────────
def befund_aus_faktenset(fs: dict) -> str:
    """
    Erzeugt den Befundtext deterministisch aus dem Faktum-Set, ohne Sprachmodell.

    Zweck ist nicht, das Sprachmodell zu ersetzen — der Text ist bewusst
    schematisch. Er dient als

      1. Referenz für die Evaluation: Jede Aussage ist per Konstruktion durch
         einen Messwert gedeckt. Weicht ein erzeugter Befund inhaltlich davon
         ab, ist das ein Prüfpunkt.
      2. Rückfallebene, wenn kein Sprachmodell verfügbar ist (etwa in einer
         Umgebung ohne externe Schnittstelle).
      3. Grundlage für die Fallabbildungen.

    Die sprachliche Glättung, die das Sprachmodell leistet, fehlt hier
    absichtlich.
    """
    feat = fs.get("classical_features_verwertbar") or fs.get("classical_features", {})
    m = fs.get("morphologie", {})
    rh, lag = m.get("rhythmus", {}), m.get("lagetyp", {})
    q, rp = m.get("q_zacken", {}), m.get("r_progression", {})
    st, tw = m.get("st_strecke", {}), m.get("t_wellen", {})

    hr = feat.get("hr_bpm")
    saetze = []

    # 1. Rhythmus, Frequenz, Lagetyp
    rhythmus = "Sinusrhythmus" if rh.get("regelmaessig") else "unregelmäßiger Rhythmus"
    teil = f"{_gross(frequenz_attribut(hr))}er {rhythmus}" \
        if hr is not None else rhythmus.capitalize()
    if hr is not None:
        teil += f", durchschnittliche Frequenz {hr:.0f}/min"
    if lag.get("beurteilbar"):
        teil += f", {lag.get('lagetyp')}"
    saetze.append(teil + ".")

    # 2. Zeiten
    zeiten, unauffaellig = [], []
    for name, wert, unter, ober, ober_wort in [
            ("PQ", feat.get("pr_interval_ms"), 120, 200, "verlängert"),
            ("QRS", feat.get("qrs_duration_ms"), None, 120, "verbreitert"),
            ("QTc", feat.get("qtc_bazett_ms"), 350, 440, "verlängert")]:
        if wert is None:
            zeiten.append(f"{name}-Zeit nicht verwertbar")
        else:
            b = _bewertung(wert, unter, ober, name_ober=ober_wort)
            if b == "normwertig":
                unauffaellig.append(name)
            else:
                zeiten.append(f"{name}-Zeit mit {wert:.0f} ms {b}")
    if len(unauffaellig) == 3:
        saetze.append("PQ-, QRS- sowie QTc-Zeiten normwertig.")
    else:
        if unauffaellig:
            zeiten.append(("-, ".join(unauffaellig)) + "-Zeiten normwertig"
                          if len(unauffaellig) > 1
                          else f"{unauffaellig[0]}-Zeit normwertig")
        saetze.append(_gross(", ".join(zeiten)) + ".")

    # 3. Q-Zacken
    if not q.get("beurteilbar"):
        saetze.append("Q-Zacken nicht beurteilbar.")
    elif q.get("pardee_q_vorhanden"):
        ltg = ", ".join(q.get("pathologische_q_ableitungen", []))
        grp = " und ".join(q.get("betroffene_gruppen", []))
        saetze.append(f"Pathologische Q-Zacken in {ltg} ({grp}).")
    else:
        saetze.append("Kein Pardée-Q.")

    # 4. R-Progression
    if rp.get("beurteilbar"):
        t = []
        pn = rp.get("r_progression_normal")
        t.append("normale R-Progredienz" if pn else
                 "verzögerte R-Progredienz" if pn is not None else
                 "R-Progredienz nicht beurteilbar")
        if rp.get("rs_umschlag"):
            t.append(f"R/S-Umschlag in {rp['rs_umschlag']}")
        t.append("S-Persistenz in V6" if rp.get("s_persistenz_v6")
                 else "keine S-Persistenz in V6")
        saetze.append(_gross(", ".join(t)) + ".")

    # 5. ST-Strecke
    if not st.get("beurteilbar"):
        saetze.append("ST-Strecke nicht beurteilbar.")
    elif st.get("relevante_hebung") or st.get("relevante_senkung"):
        t = []
        if st.get("relevante_hebung"):
            t.append(f"ST-Hebungen in {', '.join(st.get('hebungen', []))}")
        if st.get("relevante_senkung"):
            t.append(f"ST-Senkungen in {', '.join(st.get('senkungen', []))}")
        saetze.append(_gross("; ".join(t)) + ".")
    else:
        saetze.append("Keine relevanten ST-Hebungen oder -Senkungen.")

    # 6. T-Wellen
    if not tw.get("beurteilbar"):
        saetze.append("T-Wellen nicht beurteilbar.")
    elif tw.get("relevante_negativierung"):
        saetze.append("T-Negativierungen in "
                      f"{', '.join(tw.get('pathologisch_negativ', []))}.")
    else:
        saetze.append("Keine T-Negativierungen.")

    # 7. Extrasystolen
    n_es = rh.get("extrasystolen_anzahl")
    if n_es is None:
        saetze.append("Extrasystolen nicht beurteilbar.")
    elif n_es == 0:
        saetze.append("Keine Extrasystolen.")
    else:
        typ = rh.get("extrasystolen_typ") or ""
        wort = "Eine Extrasystole" if n_es == 1 else f"{n_es} Extrasystolen"
        saetze.append(f"{wort}{', ' + typ if typ else ''}.")

    befund = " ".join(saetze)

    # 8. Beurteilung
    diag = fs.get("diagnose", {})
    beurteilung = " ".join(f"{f}." for f in diag.get("formulierungen", []))
    text = f"{befund}\n\nBeurteilung: {beurteilung}"

    einschraenkungen = []
    if diag.get("vorbehalt"):
        einschraenkungen.append(diag["vorbehalt"])
    art = fs.get("plausibility", {}).get("artefakte", [])
    if art:
        einschraenkungen.append(
            f"{len(art)} Messwert(e) als Artefakt verworfen und nicht beurteilt.")
    if einschraenkungen:
        text += "\n\nEinschränkung: " + " ".join(einschraenkungen)
    return text
