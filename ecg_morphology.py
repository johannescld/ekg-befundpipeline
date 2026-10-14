"""
Morphologische EKG-Analyse über alle 12 Ableitungen.

Erweitert das Faktum-Set um die Messgrößen, die ein lehrbuchkonformer
Befund braucht und die bisher fehlten:

  - Rhythmus (Sinusrhythmus, Regelmäßigkeit, RR-Variabilität)
  - Lagetyp über die elektrische Herzachse (Ableitung I und aVF)
  - Q-Zacken / Pardée-Q je Ableitungsgruppe
  - R-Progression, R/S-Umschlag, S-Persistenz in V6
  - ST-Strecken-Hebungen und -Senkungen (J-Punkt + 60 ms)
  - T-Negativierungen
  - Extrasystolen (vorzeitige Schläge, SVES/VES-Unterscheidung)

Methodik
--------
Alle Amplitudenmessungen erfolgen auf einem **Median-Schlag** je Ableitung
(Beat-Averaging): Fenster um jeden R-Peak, elementweiser Median über alle
Schläge. Das unterdrückt Rauschen und Einzelartefakte deutlich besser als
die Messung an einem beliebigen Einzelschlag.

Die Fiducial-Punkte (QRS-Onset/-Offset, T-Offset) werden global aus der
Ableitung II bestimmt und als Offsets relativ zum R-Peak auf alle
Ableitungen übertragen — übliche Praxis, da Intervalle global definiert
sind, Amplituden aber ableitungsspezifisch.

Jede Messgröße wird als `None` zurückgegeben, wenn sie nicht sicher
bestimmbar ist. Es wird nie geraten — ein fehlender Wert ist für die
nachgelagerte Befundgenerierung verwertbar, ein falscher nicht.

Grenzwerte: siehe REFERENZ-Konstanten. Diese sind gängige klinische
Konventionen und müssen in der Dissertation belegt werden (offener Punkt).
"""

import warnings

import numpy as np
import neurokit2 as nk

from ekg_config import BASE_STR  # zentrale Pfadkonfiguration


warnings.filterwarnings("ignore")

# Ableitungsreihenfolge PTB-XL / Standard-12-Kanal
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_IDX = {name: i for i, name in enumerate(LEADS)}

# Klinisch zusammenhängende Ableitungsgruppen
LEAD_GROUPS = {
    "inferior":       ["II", "III", "aVF"],
    "lateral":        ["I", "aVL", "V5", "V6"],
    "anteroseptal":   ["V1", "V2", "V3", "V4"],
}

# ─── Referenzwerte (klinische Konvention — in der Arbeit zu belegen) ──────────
Q_MIN_DURATION_MS = 40      # Pardée-Q: Dauer >= 40 ms
Q_MIN_RATIO       = 0.25    # und Tiefe > 25 % der folgenden R-Zacke
Q_MIN_AMPLITUDE_MV = 0.10   # darunter nicht als Q-Zacke beurteilt
ST_ELEV_MV        = 0.10    # ST-Hebung ab 0.1 mV (Extremitäten + V4-V6)
ST_ELEV_MV_V2V3_M = 0.20    # V2/V3 Männer >= 40 J.
ST_ELEV_MV_V2V3_W = 0.15    # V2/V3 Frauen
ST_DEPR_MV        = -0.10   # ST-Senkung ab -0.1 mV
T_NEG_MV          = -0.10   # T-Negativierung ab -0.1 mV
S_PERSIST_V6_MV   = 0.10    # S-Persistenz in V6 ab 0.1 mV Tiefe
R_V3_MIN_MV       = 0.30    # darunter: verzögerte R-Progression (poor R wave progr.)
QRS_FRAC_WINDOW   = 0.15    # Schwelle für das Amplituden-Messfenster
QRS_FRAC_DURATION = 0.08    # Schwelle für die QRS-Dauerangabe (s. _qrs_bounds_global)
PVC_PREMATURITY   = 0.80    # RR < 80 % des Median-RR = vorzeitig
PVC_QRS_WIDE_MS   = 120     # breiter Kammerkomplex => ventrikulär
RR_IRREGULAR_CV   = 0.15    # Variationskoeffizient der RR-Intervalle

# Ableitungen, in denen eine negative T-Welle physiologisch ist
T_NEG_PHYSIOLOGIC = {"aVR", "V1", "III"}


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────
def _safe(val, digits=2):
    """Rundet, gibt None zurück wenn nicht endlich."""
    if val is None:
        return None
    v = float(val)
    return round(v, digits) if np.isfinite(v) else None


def _median_beat(signal, r_peaks, sr, pre_ms=250, post_ms=450):
    """
    Median-Schlag je Ableitung.

    Rückgabe: (beats, r_offset) mit beats.shape = (n_samples, n_leads);
    r_offset = Index des R-Peaks innerhalb des Fensters.
    """
    pre, post = int(pre_ms * sr / 1000), int(post_ms * sr / 1000)
    windows = []
    for r in r_peaks:
        a, b = int(r) - pre, int(r) + post
        if a < 0 or b > signal.shape[0]:
            continue
        windows.append(signal[a:b])
    if len(windows) < 2:
        return None, None
    return np.median(np.stack(windows), axis=0), pre


def _baseline(beat, r_off, qrs_on_off, sr):
    """
    Isoelektrische Linie je Ableitung aus dem PQ-Segment
    (20 ms vor QRS-Beginn). Fallback: Median des ganzen Schlags.
    """
    if qrs_on_off is None:
        return np.median(beat, axis=0)
    start = r_off + qrs_on_off - int(30 * sr / 1000)
    end = r_off + qrs_on_off - int(5 * sr / 1000)
    if start < 0 or end <= start:
        return np.median(beat, axis=0)
    return beat[start:end].mean(axis=0)


def _decompose_qrs(seg, sr, r_min_mv=0.10, neg_min_mv=0.05):
    """
    Zerlegt einen baseline-korrigierten QRS-Komplex in Q-, R- und S-Zacke.

    Die Unterscheidung ist rein zeitlich definiert und darf nicht über
    Extremwerte abgekürzt werden:
        Q = negative Auslenkung VOR der R-Zacke
        R = erste positive Auslenkung >= r_min_mv
        S = negative Auslenkung NACH der R-Zacke
    Fehlt jede positive Auslenkung, liegt ein QS-Komplex vor; dessen
    Negativbewegung ist dann eine Q-Zacke, keine S-Zacke.

    Q-Dauer wird vom Unterschreiten der Schwelle bis zur Rückkehr zur
    Nulllinie gemessen (nicht bis zur ersten Positivbewegung) — sonst
    zählt bei rS-Komplexen die gesamte S-Zacke mit.

    Rückgabe: dict(q_amp, q_dur_ms, r_amp, s_amp, muster)
    """
    n = len(seg)
    if n == 0:
        return None

    pos_idx = np.where(seg >= r_min_mv)[0]

    if len(pos_idx) == 0:
        # QS-Komplex: rein negativer Kammerkomplex
        q_amp = float(seg.min())
        q_dur = float(np.sum(seg <= -neg_min_mv) / sr * 1000)
        return {"q_amp": q_amp, "q_dur_ms": q_dur, "r_amp": 0.0,
                "s_amp": 0.0, "muster": "QS"}

    r_start = int(pos_idx[0])
    r_idx = int(r_start + np.argmax(seg[r_start:]))
    r_amp = float(seg[r_idx])

    # ── Q-Zacke: nur der Abschnitt vor Beginn der R-Zacke ────────────────
    pre = seg[:r_start]
    if len(pre) and pre.min() <= -neg_min_mv:
        q_amp = float(pre.min())
        below = np.where(pre <= -neg_min_mv)[0]
        q_dur = float((below[-1] - below[0] + 1) / sr * 1000)
    else:
        q_amp, q_dur = 0.0, 0.0

    # ── S-Zacke: nach dem R-Gipfel ───────────────────────────────────────
    post = seg[r_idx:]
    s_amp = float(abs(min(post.min(), 0.0))) if len(post) else 0.0

    muster = ("qRs" if q_amp < 0 and s_amp > neg_min_mv else
              "qR" if q_amp < 0 else
              "rS" if s_amp > abs(r_amp) else "Rs")

    return {"q_amp": q_amp, "q_dur_ms": q_dur, "r_amp": r_amp,
            "s_amp": s_amp, "muster": muster}


def _qrs_bounds_global(beat, r_off, base, sr, frac=0.15):
    """
    QRS-Grenzen aus dem Betrag über alle 12 Ableitungen.

    Die QRS-Dauer ist klinisch global definiert: vom frühesten Beginn der
    Kammererregung in irgendeiner Ableitung bis zum spätesten Ende. Eine
    Bestimmung allein aus Ableitung II unterschätzt sie systematisch — bei
    Schenkelblöcken liegt der späte Anteil oft nur in einzelnen Ableitungen.

    Hier wird die über alle Ableitungen summierte absolute Auslenkung
    (räumlicher Betrag) gebildet und der zusammenhängende Bereich um den
    R-Peak bestimmt, in dem sie 'frac' des Maximums überschreitet.

    Der Schwellenanteil 'frac' ist gegen die Ground Truth kalibriert
    (40 CLBBB-, 40 CRBBB-, 40 NORM-EKGs aus Fold 10):

        frac   CLBBB-Median   CRBBB-Median   NORM-Median   NORM ≥120 ms
        0.05      292 ms         260 ms         89 ms          30 %
        0.08      149 ms         148 ms         70 ms           0 %
        0.15      110 ms         114 ms         56 ms           0 %

    QRS_FRAC_WINDOW = 0.15 begrenzt das Fenster für die Amplitudenmessung
    (eng, damit keine ST-Anteile hineinragen); QRS_FRAC_DURATION = 0.08
    dient der Dauerangabe, weil der niederamplitudige Endteil eines
    Schenkelblocks sonst abgeschnitten wird. Siehe validate_morphology.py.

    Rückgabe: (onset, offset) als Offsets relativ zum R-Peak, oder None.
    """
    v = np.sum(np.abs(beat - base), axis=1)

    # Suchfenster: -150 ms bis +250 ms um den R-Peak
    lo = max(0, r_off - int(150 * sr / 1000))
    hi = min(len(v), r_off + int(250 * sr / 1000))
    if hi - lo < 10:
        return None

    seg = v[lo:hi]
    peak = int(np.argmax(seg))
    thr = seg[peak] * frac
    if not np.isfinite(thr) or thr <= 0:
        return None

    i = peak
    while i > 0 and seg[i] > thr:
        i -= 1
    j = peak
    while j < len(seg) - 1 and seg[j] > thr:
        j += 1

    on, off = lo + i - r_off, lo + j - r_off
    dur_ms = (off - on) / sr * 1000
    # physiologische Plausibilität: 50-250 ms
    if not (50 <= dur_ms <= 250):
        return None
    return int(on), int(off)


def _relative_fiducials(waves, r_peaks, sr):
    """
    Mittlere Offsets der Fiducial-Punkte relativ zum R-Peak (in Samples).
    Rückgabe: dict mit qrs_onset, qrs_offset, t_offset, t_peak (oder None).
    """
    out = {}
    keys = {"qrs_onset": "ECG_R_Onsets", "qrs_offset": "ECG_R_Offsets",
            "t_offset": "ECG_T_Offsets", "t_peak": "ECG_T_Peaks"}
    r = np.asarray(r_peaks, dtype=float)
    for name, key in keys.items():
        pts = waves.get(key)
        if not pts:
            out[name] = None
            continue
        deltas = []
        for p in pts:
            if p is None or not np.isfinite(p):
                continue
            # nächstgelegener R-Peak
            j = int(np.argmin(np.abs(r - p)))
            d = p - r[j]
            # nur plausible Zuordnungen
            if -0.25 * sr < d < 0.6 * sr:
                deltas.append(d)
        out[name] = int(np.median(deltas)) if deltas else None
    return out


# ─── Einzelanalysen ──────────────────────────────────────────────────────────
def analyze_rhythm(r_peaks, sr, qrs_ms=None):
    """Rhythmus, Regelmäßigkeit, Extrasystolen."""
    if len(r_peaks) < 3:
        return {"beurteilbar": False, "grund": "zu wenige R-Zacken"}

    rr = np.diff(np.asarray(r_peaks, dtype=float)) / sr * 1000   # ms
    rr_med = float(np.median(rr))
    cv = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else None

    # vorzeitige Schläge: deutlich kürzeres RR als der Median
    premature = [i for i, v in enumerate(rr) if v < PVC_PREMATURITY * rr_med]

    # kompensatorische Pause: auf ein kurzes RR folgt ein langes
    compensated = sum(1 for i in premature
                      if i + 1 < len(rr) and rr[i + 1] > 1.15 * rr_med)

    typ = None
    if premature:
        typ = ("ventrikulär (breiter Kammerkomplex)"
               if qrs_ms and qrs_ms >= PVC_QRS_WIDE_MS
               else "supraventrikulär (schmaler Kammerkomplex)")

    return {
        "beurteilbar": True,
        "hr_mean_bpm": _safe(60000 / rr_med, 0),
        "rr_median_ms": _safe(rr_med, 0),
        "rr_variationskoeffizient": _safe(cv, 3),
        "regelmaessig": bool(cv is not None and cv < RR_IRREGULAR_CV),
        "extrasystolen_anzahl": len(premature),
        "extrasystolen_mit_kompensatorischer_pause": compensated,
        "extrasystolen_typ": typ,
    }


def analyze_axis(beat, r_off, base, fid, sr):
    """Elektrische Herzachse und Lagetyp aus der QRS-Nettoamplitude."""
    on, off = fid.get("qrs_onset"), fid.get("qrs_offset")
    if on is None or off is None:
        return {"beurteilbar": False, "grund": "QRS-Grenzen nicht bestimmbar"}

    a, b = r_off + on, r_off + off
    if a < 0 or b > beat.shape[0] or b <= a:
        return {"beurteilbar": False, "grund": "QRS-Fenster außerhalb des Schlags"}

    # Nettofläche des QRS-Komplexes (algebraisch, baseline-korrigiert)
    net = {}
    for name in ("I", "aVF"):
        i = LEAD_IDX[name]
        seg = beat[a:b, i] - base[i]
        net[name] = float(np.sum(seg))

    if abs(net["I"]) < 1e-9 and abs(net["aVF"]) < 1e-9:
        return {"beurteilbar": False, "grund": "QRS-Amplituden zu klein"}

    axis = float(np.degrees(np.arctan2(net["aVF"], net["I"])))

    # gängige deutsche Einteilung (Cabrera-Kreis)
    if -90 <= axis < -30:
        typ = "überdrehter Linkstyp"
    elif -30 <= axis < 30:
        typ = "Linkstyp"
    elif 30 <= axis < 60:
        typ = "Indifferenztyp"
    elif 60 <= axis < 90:
        typ = "Steiltyp"
    elif 90 <= axis < 120:
        typ = "Rechtstyp"
    elif 120 <= axis <= 180:
        typ = "überdrehter Rechtstyp"
    else:
        typ = "überdrehter Rechtstyp (nordwestliche Achse)"

    return {"beurteilbar": True, "achse_grad": _safe(axis, 0), "lagetyp": typ}


def analyze_q_waves(beat, r_off, base, fid, sr, axis_deg=None):
    """
    Pathologische Q-Zacken (Pardée-Q) je Ableitung.

    axis_deg steuert die lagetypabhängige Ausnahme: bei steiler oder
    rechtsgerichteter Herzachse ist ein QS- oder rS-Komplex in aVL
    physiologisch, bei linksgerichteter Achse ein Q in III. Solche
    Q-Zacken werden erfasst, aber nicht als pathologisch gewertet.
    """
    on, off = fid.get("qrs_onset"), fid.get("qrs_offset")
    if on is None or off is None:
        return {"beurteilbar": False, "grund": "QRS-Grenzen nicht bestimmbar"}

    a, b = r_off + on, r_off + off
    if a < 0 or b > beat.shape[0] or b <= a:
        return {"beurteilbar": False, "grund": "QRS-Fenster außerhalb des Schlags"}

    pathologic, details = [], {}
    for name in LEADS:
        if name == "aVR":          # in aVR ist ein Q ohne Aussagekraft
            continue
        i = LEAD_IDX[name]
        d = _decompose_qrs(beat[a:b, i] - base[i], sr)
        if d is None:
            continue

        q_amp, q_dur_ms = d["q_amp"], d["q_dur_ms"]
        if d["muster"] == "QS":
            # QS-Komplex gilt in den Brustwandableitungen als Infarktäquivalent
            is_path = True
        elif q_amp >= -Q_MIN_AMPLITUDE_MV:
            is_path = False        # zu flach, um beurteilt zu werden
        else:
            # Pardée-Q: ausreichende Dauer UND relative Tiefe.
            # Die Dauer ist das tragende Kriterium — eine tiefe, aber sehr
            # kurze Q-Zacke bei kleiner R-Amplitude (häufig in III/aVF) ist
            # kein Infarktzeichen.
            is_path = (q_dur_ms >= Q_MIN_DURATION_MS and
                       (d["r_amp"] <= 0.05 or
                        abs(q_amp) > Q_MIN_RATIO * d["r_amp"]))

        if q_amp < 0 or d["muster"] == "QS":
            details[name] = {"q_amplitude_mv": _safe(q_amp),
                             "q_dauer_ms": _safe(q_dur_ms, 0),
                             "r_amplitude_mv": _safe(d["r_amp"]),
                             "muster": d["muster"],
                             "lagetypbedingt": False,
                             "pathologisch": is_path}
        if is_path:
            pathologic.append(name)

    # ── Lagetypabhängige Ausnahmen ──────────────────────────────────────────
    # Ein QS- oder rS-Komplex in aVL ist bei steiler Achse physiologisch,
    # ebenso ein Q in III bei linksgerichteter Achse. Beides gilt aber NUR
    # für die ISOLIERTE Ableitung: Sind die übrigen Ableitungen derselben
    # Gruppe mitbetroffen (etwa II und aVF bei inferiorem Infarkt), liegt
    # keine Lagevariante vor und die Q-Zacke bleibt pathologisch.
    if axis_deg is not None:
        for name, partner, bedingung in [
                ("aVL", ["I", "V5", "V6"], axis_deg >= 60),
                ("III", ["II", "aVF"], axis_deg <= 0)]:
            if not bedingung or name not in pathologic:
                continue
            if any(p in pathologic for p in partner):
                continue        # Gruppe mitbetroffen -> echte Pathologie
            pathologic.remove(name)
            if name in details:
                details[name]["pathologisch"] = False
                details[name]["lagetypbedingt"] = True

    groups = [g for g, leads in LEAD_GROUPS.items()
              if sum(1 for l in leads if l in pathologic) >= 2]

    return {
        "beurteilbar": True,
        "pathologische_q_ableitungen": pathologic,
        "betroffene_gruppen": groups,      # >= 2 Ableitungen einer Gruppe
        "pardee_q_vorhanden": bool(groups),
        "details": details,
    }


def analyze_r_progression(beat, r_off, base, fid, sr):
    """R-Progression, R/S-Umschlag und S-Persistenz in V6."""
    on, off = fid.get("qrs_onset"), fid.get("qrs_offset")
    if on is None or off is None:
        return {"beurteilbar": False, "grund": "QRS-Grenzen nicht bestimmbar"}

    a, b = r_off + on, r_off + off
    if a < 0 or b > beat.shape[0] or b <= a:
        return {"beurteilbar": False, "grund": "QRS-Fenster außerhalb des Schlags"}

    r_amp, s_amp, ratio, muster = {}, {}, {}, {}
    for name in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        i = LEAD_IDX[name]
        d = _decompose_qrs(beat[a:b, i] - base[i], sr)
        if d is None:
            r_amp[name] = s_amp[name] = ratio[name] = muster[name] = None
            continue
        # S = Negativbewegung NACH der R-Zacke; bei QS-Komplex gibt es
        # definitionsgemäß keine S-Zacke (die Negativität ist die Q-Zacke).
        r, s = float(d["r_amp"]), float(d["s_amp"])
        r_amp[name] = _safe(r)
        s_amp[name] = _safe(s)
        ratio[name] = _safe(r / s, 2) if s > 0.01 else None
        muster[name] = d["muster"]

    # R/S-Umschlag: erste Brustwandableitung mit R > S
    transition = None
    for name in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        if r_amp[name] is not None and s_amp[name] is not None \
                and r_amp[name] > s_amp[name]:
            transition = name
            break

    # R-Progression nach der klinischen Definition der "poor R wave
    # progression": R in V3 mindestens 0.3 mV und Zunahme von V1 nach V4.
    # Eine monotone Zunahme bis V4 zu fordern wäre falsch — das R-Maximum
    # liegt physiologisch in V3 bis V5 und fällt danach wieder ab.
    r_v1, r_v3, r_v4 = r_amp.get("V1"), r_amp.get("V3"), r_amp.get("V4")
    if r_v3 is None or r_v4 is None or r_v1 is None:
        progression_normal = None
    else:
        progression_normal = bool(r_v3 >= R_V3_MIN_MV and r_v4 > r_v1)

    s_v6 = s_amp.get("V6")
    return {
        "beurteilbar": True,
        "r_amplituden_mv": r_amp,
        "s_amplituden_mv": s_amp,
        "rs_verhaeltnis": ratio,
        "qrs_muster": muster,
        "rs_umschlag": transition,
        "rs_umschlag_normal": transition in ("V3", "V4"),
        "r_progression_normal": bool(progression_normal),
        "s_persistenz_v6": bool(s_v6 is not None and s_v6 > S_PERSIST_V6_MV),
        "s_amplitude_v6_mv": s_v6,
    }


def analyze_st_segment(beat, r_off, base, fid, sr, sex=None, age=None):
    """ST-Strecken-Abweichung, gemessen 60 ms nach dem J-Punkt."""
    off = fid.get("qrs_offset")
    if off is None:
        return {"beurteilbar": False, "grund": "QRS-Ende (J-Punkt) nicht bestimmbar"}

    j = r_off + off
    idx = j + int(60 * sr / 1000)
    if idx >= beat.shape[0] or j < 0:
        return {"beurteilbar": False, "grund": "ST-Messpunkt außerhalb des Schlags"}

    elev_thr_v2v3 = (ST_ELEV_MV_V2V3_W if sex == "weiblich"
                     else ST_ELEV_MV_V2V3_M)

    dev, elevations, depressions = {}, [], []
    for name in LEADS:
        i = LEAD_IDX[name]
        d = float(beat[idx, i] - base[i])
        dev[name] = _safe(d)
        thr_up = elev_thr_v2v3 if name in ("V2", "V3") else ST_ELEV_MV
        if d >= thr_up:
            elevations.append(name)
        elif d <= ST_DEPR_MV:
            depressions.append(name)

    groups_elev = [g for g, leads in LEAD_GROUPS.items()
                   if sum(1 for l in leads if l in elevations) >= 2]
    groups_depr = [g for g, leads in LEAD_GROUPS.items()
                   if sum(1 for l in leads if l in depressions) >= 2]

    return {
        "beurteilbar": True,
        "messpunkt": "J-Punkt + 60 ms",
        "abweichung_mv": dev,
        "hebungen": elevations,
        "senkungen": depressions,
        "hebung_gruppen": groups_elev,
        "senkung_gruppen": groups_depr,
        "relevante_hebung": bool(groups_elev),
        "relevante_senkung": bool(groups_depr),
    }


def analyze_t_waves(beat, r_off, base, fid, sr):
    """T-Wellen-Polarität und Negativierungen."""
    tp, off = fid.get("t_peak"), fid.get("qrs_offset")
    if tp is None:
        if off is None:
            return {"beurteilbar": False, "grund": "T-Welle nicht bestimmbar"}
        tp = off + int(150 * sr / 1000)      # Näherung

    idx = r_off + tp
    if idx < 0 or idx >= beat.shape[0]:
        return {"beurteilbar": False, "grund": "T-Messpunkt außerhalb des Schlags"}

    amps, negative = {}, []
    for name in LEADS:
        i = LEAD_IDX[name]
        v = float(beat[idx, i] - base[i])
        amps[name] = _safe(v)
        if v <= T_NEG_MV:
            negative.append(name)

    pathologic = [n for n in negative if n not in T_NEG_PHYSIOLOGIC]
    groups = [g for g, leads in LEAD_GROUPS.items()
              if sum(1 for l in leads if l in pathologic) >= 2]

    return {
        "beurteilbar": True,
        "t_amplituden_mv": amps,
        "negative_ableitungen": negative,
        "pathologisch_negativ": pathologic,   # ohne aVR/V1/III
        "betroffene_gruppen": groups,
        "relevante_negativierung": bool(pathologic),
    }


# ─── Gesamtanalyse ───────────────────────────────────────────────────────────
def analyze_morphology(signal: np.ndarray, sr: int = 500,
                       sex: str = None, age: int = None,
                       qrs_ms: float = None) -> dict:
    """
    Vollständige morphologische Analyse eines 12-Kanal-EKGs.

    signal: (n_samples, 12) in mV, Reihenfolge I, II, III, aVR, aVL, aVF, V1-V6
    Rückgabe: verschachteltes dict, jede Sektion mit 'beurteilbar'-Flag.
    """
    result = {"methodik": {
        "amplituden": "Median-Schlag über alle Herzaktionen (Beat-Averaging)",
        "nulllinie": "PQ-Segment (30-5 ms vor QRS-Beginn)",
        "st_messpunkt": "J-Punkt + 60 ms",
    }}

    if signal.ndim != 2 or signal.shape[1] != 12:
        return {"beurteilbar": False,
                "grund": f"12 Ableitungen erwartet, {signal.shape} erhalten"}

    # R-Zacken und Fiducial-Punkte aus Ableitung II
    try:
        lead_ii = signal[:, LEAD_IDX["II"]]
        _, info = nk.ecg_process(lead_ii, sampling_rate=sr)
        r_peaks = [int(p) for p in info.get("ECG_R_Peaks", []) if np.isfinite(p)]
        waves = nk.ecg_delineate(lead_ii, info, sampling_rate=sr, method="dwt")[1]
    except Exception as e:
        return {"beurteilbar": False, "grund": f"Signalanalyse fehlgeschlagen: {e}"}

    if len(r_peaks) < 3:
        return {"beurteilbar": False, "grund": "zu wenige detektierte Herzaktionen"}

    result["rhythmus"] = analyze_rhythm(r_peaks, sr, qrs_ms)

    beat, r_off = _median_beat(signal, r_peaks, sr)
    if beat is None:
        result["beurteilbar"] = False
        result["grund"] = "kein vollständiger Median-Schlag bildbar"
        return result

    fid = _relative_fiducials(waves, r_peaks, sr)
    base = _baseline(beat, r_off, fid.get("qrs_onset"), sr)

    # Enges Fenster (0.15) für alle Amplitudenmessungen; die Delineation
    # aus Ableitung II ist nur Rückfallebene.
    bounds = _qrs_bounds_global(beat, r_off, base, sr, frac=QRS_FRAC_WINDOW)
    if bounds is not None:
        fid["qrs_onset"], fid["qrs_offset"] = bounds
        fid["qrs_quelle"] = "global (12 Ableitungen)"
        base = _baseline(beat, r_off, fid["qrs_onset"], sr)
    else:
        fid["qrs_quelle"] = "Ableitung II (dwt)"

    result["fiducial_offsets_samples"] = fid

    # Weites Fenster (0.08) allein für die Dauerangabe — sonst wird der
    # niederamplitudige Endteil eines Schenkelblocks abgeschnitten.
    dur_bounds = _qrs_bounds_global(beat, r_off, base, sr,
                                    frac=QRS_FRAC_DURATION)
    qrs_fid = (_safe((dur_bounds[1] - dur_bounds[0]) / sr * 1000, 0)
               if dur_bounds is not None else None)
    result["intervalle_aus_fiducials"] = {
        "qrs_dauer_ms": qrs_fid,
        "hinweis": ("gegen CLBBB/CRBBB kalibriert; unterschätzt schmale "
                    "Komplexe systematisch um ca. 20 ms — für den Befundtext "
                    "ist der Wert aus build_fact_set massgeblich"),
    }
    if qrs_ms is None and qrs_fid is not None:
        result["rhythmus"] = analyze_rhythm(r_peaks, sr, qrs_fid)

    result["lagetyp"] = analyze_axis(beat, r_off, base, fid, sr)
    axis_deg = result["lagetyp"].get("achse_grad")

    # Q-Zacken am WEITEN Fenster messen: Die Q-Zacke liegt unmittelbar am
    # Beginn des Kammerkomplexes. Das enge Amplitudenfenster setzt den
    # QRS-Beginn systematisch zu spät an und verkürzt die gemessene Q-Dauer,
    # sodass das 40-ms-Kriterium selbst bei echten Infarkt-Q-Zacken verfehlt
    # wird (nachgewiesen an EKG 175, Hinterwandinfarkt mit QS in III/aVF).
    fid_q = dict(fid)
    if dur_bounds is not None:
        fid_q["qrs_onset"], fid_q["qrs_offset"] = dur_bounds
    result["q_zacken"] = analyze_q_waves(beat, r_off, base, fid_q, sr, axis_deg)
    result["r_progression"] = analyze_r_progression(beat, r_off, base, fid, sr)
    result["st_strecke"] = analyze_st_segment(beat, r_off, base, fid, sr, sex, age)
    result["t_wellen"] = analyze_t_waves(beat, r_off, base, fid, sr)
    result["beurteilbar"] = True
    return result


# ─── Selbsttest ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import pandas as pd

    DATAFOLDER = (BASE_STR +
                  "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/")

    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    X = np.load(DATAFOLDER + "raw500_signals.npy", mmap_mode="r")
    pos = {e: i for i, e in enumerate(Y.index)}

    for eid in [9, 65, 175, 128, 271, 7670]:
        row = Y.loc[eid]
        sig = np.asarray(X[pos[eid]], dtype=float)
        sex = "weiblich" if row.sex == 1 else "männlich"   # PTB-XL: 0 = m, 1 = w
        res = analyze_morphology(sig, sr=500, sex=sex, age=row.age)
        print(f"\n{'='*70}\nEKG {eid}  ({sex}, {row.age} J.)\n{'='*70}")
        if not res.get("beurteilbar"):
            print("  nicht beurteilbar:", res.get("grund"))
            continue
        print(f"  Rhythmus:      {res['rhythmus'].get('hr_mean_bpm')} /min, "
              f"regelmäßig={res['rhythmus'].get('regelmaessig')}, "
              f"ES={res['rhythmus'].get('extrasystolen_anzahl')}")
        print(f"  Lagetyp:       {res['lagetyp'].get('lagetyp')} "
              f"({res['lagetyp'].get('achse_grad')}°)")
        print(f"  Pardée-Q:      {res['q_zacken'].get('pardee_q_vorhanden')} "
              f"{res['q_zacken'].get('betroffene_gruppen')}")
        print(f"  R/S-Umschlag:  {res['r_progression'].get('rs_umschlag')} "
              f"(normal={res['r_progression'].get('rs_umschlag_normal')}), "
              f"S-Persistenz V6={res['r_progression'].get('s_persistenz_v6')}")
        print(f"  ST-Hebungen:   {res['st_strecke'].get('hebungen')} "
              f"| Senkungen: {res['st_strecke'].get('senkungen')}")
        print(f"  T-negativ:     {res['t_wellen'].get('pathologisch_negativ')}")
