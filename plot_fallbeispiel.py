"""
Fallabbildungen: EKG, extrahierte Messgrössen und erzeugter Befund auf einer Seite.

Je Fall eine A4-Seite mit drei Teilen:

  oben   12-Kanal-EKG im klinischen 3x4-Layout (25 mm/s, 10 mm/mV) mit
         Millimeterraster, darunter Rhythmusstreifen (Ableitung II, 10 s)
  mitte  Median-Schlag ausgewählter Ableitungen mit den eingezeichneten
         Messpunkten: QRS-Grenzen, Nulllinie, ST-Messpunkt (J+60 ms),
         markierte pathologische Q-Zacken
  unten  Faktum-Set (Messwerte) und der daraus erzeugte Befundtext

Damit ist jede Aussage des Befundes bis zur Stelle im Signal nachvollziehbar,
an der sie gemessen wurde.

Aufruf:
    python3 plot_fallbeispiel.py                    # Standardauswahl
    python3 plot_fallbeispiel.py --ids 9 175 271    # eigene Auswahl
"""

import argparse
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from ekg_config import DATAFOLDER, OUTFOLDER
from ecg_morphology import (analyze_morphology, _median_beat, _baseline,
                            _qrs_bounds_global, LEADS, LEAD_IDX,
                            QRS_FRAC_WINDOW, QRS_FRAC_DURATION)
from befund_schema import befund_aus_faktenset

SR = 500
EKG_ROT = "#E8A0A0"
EKG_ROT_FEIN = "#F5D5D5"
KURVE = "#111111"

# 3x4-Layout: vier Zeitfenster à 2,5 s
LAYOUT = [["I", "aVR", "V1", "V4"],
          ["II", "aVL", "V2", "V5"],
          ["III", "aVF", "V3", "V6"]]


def ekg_raster(ax, t0, t1, y0, y1):
    """Millimeterraster wie auf EKG-Papier: fein 0,04 s / 0,1 mV, grob 0,2 s / 0,5 mV."""
    for x in np.arange(t0, t1 + 1e-9, 0.04):
        ax.axvline(x, color=EKG_ROT_FEIN, lw=0.3, zorder=0)
    for y in np.arange(y0, y1 + 1e-9, 0.1):
        ax.axhline(y, color=EKG_ROT_FEIN, lw=0.3, zorder=0)
    for x in np.arange(t0, t1 + 1e-9, 0.2):
        ax.axvline(x, color=EKG_ROT, lw=0.6, zorder=0)
    for y in np.arange(y0, y1 + 1e-9, 0.5):
        ax.axhline(y, color=EKG_ROT, lw=0.6, zorder=0)


def plot_ekg_12(fig, gs, sig):
    """12-Kanal-EKG im 3x4-Layout plus Rhythmusstreifen."""
    fenster = 2.5
    for zeile, gruppe in enumerate(LAYOUT):
        ax = fig.add_subplot(gs[zeile])
        for spalte, lead in enumerate(gruppe):
            t_start = spalte * fenster
            a, b = int(t_start * SR), int((t_start + fenster) * SR)
            t = np.arange(a, b) / SR
            y = sig[a:b, LEAD_IDX[lead]]
            ax.plot(t, y, color=KURVE, lw=0.7, zorder=3)
            ax.text(t_start + 0.05, 1.35, lead, fontsize=7.5,
                    fontweight="bold", va="top", zorder=4)
            if spalte:
                ax.axvline(t_start, color="#999999", lw=0.5, zorder=2)
        ekg_raster(ax, 0, 10, -1.5, 1.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # Rhythmusstreifen
    ax = fig.add_subplot(gs[3])
    t = np.arange(sig.shape[0]) / SR
    ax.plot(t, sig[:, LEAD_IDX["II"]], color=KURVE, lw=0.7, zorder=3)
    ax.text(0.05, 1.35, "II  (Rhythmus, 10 s)", fontsize=7.5,
            fontweight="bold", va="top", zorder=4)
    ekg_raster(ax, 0, 10, -1.5, 1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot_messpunkte(fig, gs, sig, res, leads):
    """Median-Schlag mit eingezeichneten Messpunkten."""
    import neurokit2 as nk
    _, info = nk.ecg_process(sig[:, LEAD_IDX["II"]], sampling_rate=SR)
    rp = [int(p) for p in info.get("ECG_R_Peaks", []) if np.isfinite(p)]
    beat, r_off = _median_beat(sig, rp, SR)
    if beat is None:
        return
    fid = res.get("fiducial_offsets_samples", {})
    base = _baseline(beat, r_off, fid.get("qrs_onset"), SR)
    weit = _qrs_bounds_global(beat, r_off, base, SR, frac=QRS_FRAC_DURATION)

    q_path = res.get("q_zacken", {}).get("pathologische_q_ableitungen", [])
    t = (np.arange(beat.shape[0]) - r_off) / SR * 1000     # ms relativ zu R

    for i, lead in enumerate(leads):
        ax = fig.add_subplot(gs[i])
        y = beat[:, LEAD_IDX[lead]] - base[LEAD_IDX[lead]]
        ax.plot(t, y, color=KURVE, lw=1.1, zorder=3)
        ax.axhline(0, color="#888888", lw=0.6, ls="--", zorder=2)

        on, off = fid.get("qrs_onset"), fid.get("qrs_offset")
        if on is not None and off is not None:
            ax.axvspan(on / SR * 1000, off / SR * 1000, color="#4C72B0",
                       alpha=0.10, zorder=1, label="QRS (Amplitudenfenster)")
        if weit is not None:
            ax.axvline(weit[0] / SR * 1000, color="#4C72B0", lw=0.8, ls=":", zorder=2)
            ax.axvline(weit[1] / SR * 1000, color="#4C72B0", lw=0.8, ls=":", zorder=2)
        if off is not None:
            st_x = (off + int(60 * SR / 1000)) / SR * 1000
            idx = r_off + off + int(60 * SR / 1000)
            if 0 <= idx < len(y):
                ax.plot([st_x], [y[idx]], "o", ms=5, color="#C44E52", zorder=5)
                ax.annotate("ST (J+60 ms)", (st_x, y[idx]), fontsize=6,
                            xytext=(6, 6), textcoords="offset points",
                            color="#C44E52")
        if lead in q_path:
            ax.text(0.03, 0.06, "pathologische Q-Zacke", transform=ax.transAxes,
                    fontsize=6.5, color="#C44E52", fontweight="bold")
            for s in ax.spines.values():
                s.set_color("#C44E52")
                s.set_linewidth(1.2)

        ax.set_title(lead, fontsize=8, fontweight="bold", pad=2)
        ax.set_xlim(-120, 320)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.set_ylabel("mV", fontsize=7)
        ax.set_xlabel("ms rel. R", fontsize=6.5)
        ax.grid(alpha=0.25, lw=0.4)


def faktenblock(fs, res) -> str:
    """Kompakte Darstellung der extrahierten Messgrössen."""
    feat = fs.get("classical_features_verwertbar") or fs["classical_features"]
    rh = res.get("rhythmus", {})
    lag = res.get("lagetyp", {})
    q = res.get("q_zacken", {})
    rp = res.get("r_progression", {})
    st = res.get("st_strecke", {})
    tw = res.get("t_wellen", {})

    def z(v, e=""):
        return "n. b." if v is None else f"{v:.0f}{e}"

    rs = rp.get("r_amplituden_mv", {})
    ss = rp.get("s_amplituden_mv", {})
    mu = rp.get("qrs_muster", {})
    # Bei einem QS-Komplex gibt es definitionsgemäss weder R- noch S-Zacke;
    # "0.00/0.00" wäre irreführend, deshalb wird das Muster ausgewiesen.
    rs_txt = "  ".join(
        f"{k} QS" if mu.get(k) == "QS"
        else f"{k} {rs.get(k) or 0:.2f}/{ss.get(k) or 0:.2f}"
        for k in ["V1", "V2", "V3", "V4", "V5", "V6"])

    k = fs.get("qrs_kreuzpruefung", {})
    qrs_hinweis = ""
    if k and not k.get("verwertbar"):
        zeilen = textwrap.wrap(f"verworfen: {k.get('grund')}", 74)
        zeilen.append(f"(Ableitung II {k.get('messweg_ableitung_II_ms')} ms, "
                      f"12 Ableitungen {k.get('messweg_12_ableitungen_ms')} ms)")
        qrs_hinweis = "  QRS-Prüfung   " + zeilen[0] + "\n"
        for zeile in zeilen[1:]:
            qrs_hinweis += "                " + zeile + "\n"

    return (
        f"MESSGRÖSSEN (Feature Extraction)\n"
        f"  Frequenz      {z(feat.get('hr_bpm'), '/min')}   "
        f"RR-Median {z(rh.get('rr_median_ms'), ' ms')}   "
        f"CV {rh.get('rr_variationskoeffizient')}   "
        f"Extrasystolen {rh.get('extrasystolen_anzahl')}\n"
        f"  Zeiten        PQ {z(feat.get('pr_interval_ms'), ' ms')}   "
        f"QRS {z(feat.get('qrs_duration_ms'), ' ms')}   "
        f"QT {z(feat.get('qt_interval_ms'), ' ms')}   "
        f"QTc {z(feat.get('qtc_bazett_ms'), ' ms')}\n"
        f"{qrs_hinweis}"
        f"  Herzachse     {lag.get('achse_grad')}°  →  {lag.get('lagetyp')}\n"
        f"  Q-Zacken      {', '.join(q.get('pathologische_q_ableitungen', [])) or 'keine pathologischen'}"
        f"   Gruppen: {', '.join(q.get('betroffene_gruppen', [])) or '—'}\n"
        f"  R/S (mV)      {rs_txt}\n"
        f"  R/S-Umschlag  {rp.get('rs_umschlag')}   "
        f"R-Progredienz {'regelrecht' if rp.get('r_progression_normal') else 'verzögert'}   "
        f"S-Persistenz V6 {'ja' if rp.get('s_persistenz_v6') else 'nein'}\n"
        f"  ST (J+60 ms)  Hebung: {', '.join(st.get('hebungen', [])) or '—'}   "
        f"Senkung: {', '.join(st.get('senkungen', [])) or '—'}\n"
        f"  T-Wellen      negativ in: {', '.join(tw.get('negative_ableitungen', [])) or '—'}   "
        f"davon pathologisch: {', '.join(tw.get('pathologisch_negativ', [])) or '—'}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=None)
    ap.add_argument("--factsets", default="example_fact_sets_v2.json")
    ap.add_argument("--out", default=OUTFOLDER + "fallbeispiele.pdf")
    ap.add_argument("--png", action="store_true",
                    help="zusätzlich je Fall eine PNG-Datei schreiben")
    args = ap.parse_args()

    with open(args.factsets, encoding="utf-8") as f:
        fact_sets = {fs["ecg_id"]: fs for fs in json.load(f)}
    ids = args.ids or list(fact_sets)

    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    X = np.load(DATAFOLDER + "raw500_signals.npy", mmap_mode="r")
    pos = {e: i for i, e in enumerate(Y.index)}

    with PdfPages(args.out) as pdf:
        for eid in ids:
            fs = fact_sets.get(eid)
            if fs is None:
                print(f"EKG {eid}: kein Faktum-Set, übersprungen")
                continue
            row = Y.loc[eid]
            sig = np.asarray(X[pos[eid]], dtype=float)
            sex = "weiblich" if row.sex == 1 else "männlich"
            res = analyze_morphology(sig, SR, sex, row.age)

            fig = plt.figure(figsize=(8.27, 11.69))       # A4
            gs_ekg = GridSpec(4, 1, figure=fig, left=0.04, right=0.98,
                              top=0.935, bottom=0.55, hspace=0.06)
            plot_ekg_12(fig, gs_ekg, sig)

            # Für die Detailansicht: auffällige Ableitungen bevorzugen
            q_path = res.get("q_zacken", {}).get("pathologische_q_ableitungen", [])
            wahl = (q_path + ["II", "V2", "V5"])[:3]
            gs_det = GridSpec(1, 3, figure=fig, left=0.08, right=0.98,
                              top=0.515, bottom=0.40, wspace=0.28)
            plot_messpunkte(fig, gs_det, sig, res, wahl)

            wahr = ", ".join(fs.get("true_classes", [])) or "—"
            fig.suptitle(f"EKG {eid}  ·  {sex}, {int(row.age)} Jahre  ·  "
                         f"Referenzdiagnose: {wahr}",
                         fontsize=11, fontweight="bold", y=0.972)
            fig.text(0.04, 0.952,
                     "12-Kanal-EKG, 25 mm/s · 10 mm/mV        "
                     "Median-Schlag mit Messpunkten unten",
                     fontsize=7, color="#555555")

            fig.text(0.04, 0.362, faktenblock(fs, res), fontsize=6.6,
                     family="monospace", va="top")

            befund = befund_aus_faktenset(fs)
            umbrochen = "\n".join(textwrap.fill(abs_, 108)
                                  for abs_ in befund.split("\n"))
            fig.text(0.04, 0.212, "BEFUND (regelbasiert aus dem Faktum-Set)",
                     fontsize=7.5, fontweight="bold", va="top")
            fig.text(0.04, 0.194, umbrochen, fontsize=7, va="top")

            fig.text(0.04, 0.022,
                     "Blau: QRS-Fenster der Amplitudenmessung (gepunktet: weites "
                     "Fenster für Q-Dauer und QRS-Dauer). Rot: ST-Messpunkt 60 ms "
                     "nach dem J-Punkt.\nRot umrandet: Ableitung mit pathologischer "
                     "Q-Zacke. Nulllinie gestrichelt (PQ-Segment).",
                     fontsize=6, color="#555555", va="bottom")

            pdf.savefig(fig)
            if args.png:
                fig.savefig(OUTFOLDER + f"fallbeispiel_{eid}.png", dpi=130)
            plt.close(fig)
            print(f"EKG {eid}: Seite erzeugt")

    print(f"\nGespeichert: {args.out}")


if __name__ == "__main__":
    main()
