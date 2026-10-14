"""
Sankey-Diagramm des Labelschemas (Betreuer-Wunsch).

Vier Ebenen — exakt der Datenfluss der Pipeline:

  44 diagnostische SCP-ECG-Statements
      -> 23 diagnostische Subklassen (PTB-XL-Hierarchie)
          -> 19 Zielklassen (hybrides Schema dieser Arbeit)
              -> 5 Superklassen (Fallback bei Konfidenz < 0.65, Stufe 3)

Fallzahl = Anzahl EKGs mit dem jeweiligen Label (Set-Logik je EKG, identisch
zur Zählweise in train_resnet_hybrid.py). Da ein EKG mehrere Labels tragen
kann, übersteigt die Spaltensumme die EKG-Zahl (n = 21799).

Knotenhöhen sind wurzelskaliert (∝ √n): das Verhältnis 9514 (NORM) zu 14
(2AVB) wäre linear nicht darstellbar. Die Bänder sind am Zielknoten auf die
Summe der dort eingehenden Fallzahlen normiert.

Aufruf:  python3 plot_label_sankey.py
Ausgabe: ptbxl_results/label_schema_sankey.{png,pdf}
"""

import ast
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

from label_mapping_hybrid import SUBCLASS_TO_TARGET, TARGET_TO_SUPER, TARGET_LABELS_DE

from ekg_config import BASE_STR  # zentrale Pfadkonfiguration


DATAFOLDER = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTFOLDER = BASE_STR + "ptbxl_results/"

SUPER_COLORS = {"NORM": "#4C72B0", "MI": "#C44E52", "STTC": "#DD8452",
                "CD": "#55A868", "HYP": "#8172B3"}
SUPER_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]

COL_X = [0.0, 1.0, 2.0, 3.0]      # x-Position der vier Ebenen
NODE_W = 0.075                    # Knotenbreite
GAP = 1.6                         # vertikale Lücke zwischen Knoten
MIN_H = 3.4                       # Mindesthöhe eines Knotens


def load_data():
    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(DATAFOLDER + "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1.0]

    code_info = {}          # code -> (subklasse, superklasse, beschreibung)
    for code, row in diag.iterrows():
        if pd.notna(row.diagnostic_subclass) and row.diagnostic_subclass in SUBCLASS_TO_TARGET:
            code_info[code] = (row.diagnostic_subclass,
                               row.diagnostic_class,
                               str(row.get("description", "")))

    # Fallzahl = Anzahl EKGs mit diesem Label (Set-Logik je EKG), damit die
    # Zahlen exakt denen aus train_resnet_hybrid.py / der Ergebnistabelle
    # entsprechen. Ein EKG mit zwei Codes derselben Subklasse zählt einmal.
    code_counts = Counter()
    sub_counts = Counter()
    tgt_counts = Counter()
    sup_counts = Counter()
    for codes in Y.scp_codes:
        hit = [c for c in codes if c in code_info]
        for c in set(hit):
            code_counts[c] += 1
        subs = {code_info[c][0] for c in hit}
        for s in subs:
            sub_counts[s] += 1
        tgts = {SUBCLASS_TO_TARGET[s] for s in subs}
        for t in tgts:
            tgt_counts[t] += 1
        for sp in {TARGET_TO_SUPER[t] for t in tgts}:
            sup_counts[sp] += 1

    test_counts = Counter()
    for codes in Y[Y.strat_fold == 10].scp_codes:
        for t in {SUBCLASS_TO_TARGET[code_info[c][0]] for c in codes if c in code_info}:
            test_counts[t] += 1

    counts = (code_counts, sub_counts, tgt_counts, sup_counts)
    return code_info, counts, test_counts, len(Y)


def build_nodes(code_info, counts):
    """Knoten je Ebene, nach Superklasse gruppiert und innerhalb nach n sortiert."""
    sub_of = {c: code_info[c][0] for c in code_info}
    super_of_sub = {s: TARGET_TO_SUPER[SUBCLASS_TO_TARGET[s]] for s in SUBCLASS_TO_TARGET}
    code_counts, sub_counts, tgt_counts, sup_counts = counts

    def order(keys, super_key):
        out = []
        for sup in SUPER_ORDER:
            grp = [k for k in keys if super_key(k) == sup]
            out += sorted(grp, key=lambda k: -keys[k])
        return out

    lvl = [
        order(code_counts, lambda c: super_of_sub[sub_of[c]]),
        order(sub_counts, lambda s: super_of_sub[s]),
        order(tgt_counts, lambda t: TARGET_TO_SUPER[t]),
        order(sup_counts, lambda s: s),
    ]
    cnt = [code_counts, sub_counts, tgt_counts, sup_counts]
    sup = [
        {c: super_of_sub[sub_of[c]] for c in code_counts},
        {s: super_of_sub[s] for s in sub_counts},
        {t: TARGET_TO_SUPER[t] for t in tgt_counts},
        {s: s for s in sup_counts},
    ]
    return lvl, cnt, sup, sub_of


def layout(lvl, cnt, scale):
    """
    y-Positionen (unten/oben) je Knoten.

    Höhe ∝ √Fallzahl: bei einem Verhältnis von 9514 (NORM) zu 14 (2AVB) — Faktor
    680 — wären seltene Statements linear weder sicht- noch beschriftbar. Die
    Wurzelskalierung ist in der Bildunterschrift deklariert.
    """
    pos = []
    for keys, counts in zip(lvl, cnt):
        y = 0.0
        p = {}
        for k in keys:
            h = max(np.sqrt(counts[k]) * scale, MIN_H)
            p[k] = (y, y + h)
            y += h + GAP
        pos.append(p)
    return pos


def regroup(pos, lvl, sup, extra=2.4):
    """Zusätzliche Lücke zwischen Superklassen-Blöcken einfügen."""
    out = []
    for p, keys, smap in zip(pos, lvl, sup):
        shift = 0.0
        prev = None
        q = {}
        for k in keys:
            if prev is not None and smap[k] != prev:
                shift += extra
            lo, hi = p[k]
            q[k] = (lo + shift, hi + shift)
            prev = smap[k]
        out.append(q)
    return out


def center(pos, keys):
    lo = min(pos[k][0] for k in keys)
    hi = max(pos[k][1] for k in keys)
    return lo, hi


def band(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.42):
    """Gefülltes Bezier-Band zwischen zwei Knotenkanten."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color,
                           edgecolor="none", alpha=alpha, zorder=1))


def main():
    code_info, counts, test_counts, n_ecg = load_data()
    lvl, cnt, sup, sub_of = build_nodes(code_info, counts)

    print(f"Ebenen: {len(lvl[0])} SCP-Statements -> {len(lvl[1])} Subklassen "
          f"-> {len(lvl[2])} Zielklassen -> {len(lvl[3])} Superklassen")
    print(f"Kontrolle Zielklassen-Fallzahlen (Gesamtdatensatz): "
          f"NORM={cnt[2]['NORM']}, STTC={cnt[2]['STTC']}, AMI={cnt[2]['AMI']}, "
          f"LAFB/LPFB={cnt[2]['LAFB/LPFB']}")

    # Skalierung so, dass die höchste Spalte eine feste Gesamthöhe erreicht
    total_max = max(sum(np.sqrt(v) for v in c.values()) for c in cnt)
    scale = 620.0 / total_max
    pos = regroup(layout(lvl, cnt, scale), lvl, sup)

    # Spalten vertikal zentrieren
    spans = [center(p, k) for p, k in zip(pos, lvl)]
    height = max(hi - lo for lo, hi in spans)
    for i, (p, (lo, hi)) in enumerate(zip(pos, spans)):
        off = (height - (hi - lo)) / 2 - lo
        pos[i] = {k: (a + off, b + off) for k, (a, b) in p.items()}

    fig, ax = plt.subplots(figsize=(16, 17))

    # ── Bänder zeichnen ─────────────────────────────────────────────────────
    def draw_layer(i, mapping):
        """
        mapping: key der Ebene i -> key der Ebene i+1.

        Die Bandhöhe am Zielknoten wird auf die Summe der dort tatsächlich
        eingehenden Fallzahlen normiert (nicht auf die Knotenfallzahl): bei
        Mehrfachlabeln ist die Summe der Quellen groesser als die Zielzahl,
        sonst würden die Bänder aus dem Knoten herauslaufen.
        """
        inflow = defaultdict(float)
        for k in lvl[i]:
            inflow[mapping(k)] += cnt[i][k]

        used_dst = defaultdict(float)
        for k in lvl[i]:
            tgt = mapping(k)
            n = cnt[i][k]
            h_src = pos[i][k][1] - pos[i][k][0]
            h_dst = (pos[i + 1][tgt][1] - pos[i + 1][tgt][0]) * n / inflow[tgt]
            y0 = pos[i][k][0]
            y1 = pos[i + 1][tgt][0] + used_dst[tgt]
            band(ax, COL_X[i] + NODE_W, COL_X[i + 1],
                 y0, y0 + h_src, y1, y1 + h_dst, SUPER_COLORS[sup[i][k]])
            used_dst[tgt] += h_dst

    draw_layer(0, lambda c: sub_of[c])
    draw_layer(1, lambda s: SUBCLASS_TO_TARGET[s])
    draw_layer(2, lambda t: TARGET_TO_SUPER[t])

    # ── Knoten + Beschriftung ───────────────────────────────────────────────
    label_cfg = [
        dict(fs=6.4, side="left"),
        dict(fs=8.0, side="left"),
        dict(fs=8.6, side="right"),
        dict(fs=11.0, side="right"),
    ]
    for i, keys in enumerate(lvl):
        for k in keys:
            lo, hi = pos[i][k]
            col = SUPER_COLORS[sup[i][k]]
            ax.add_patch(Rectangle((COL_X[i], lo), NODE_W, hi - lo,
                                   facecolor=col, edgecolor="white",
                                   lw=.4, zorder=3))
            cfg = label_cfg[i]
            n = cnt[i][k]
            if i == 0:
                txt = f"{k}  ({n})"
            elif i == 1:
                txt = f"{k}  ({n})"
            elif i == 2:
                txt = f"{k}  ({n} | Test {test_counts.get(k,0)})"
            else:
                txt = f"{k}   n={n}"
            if cfg["side"] == "left":
                ax.text(COL_X[i] - 0.03, (lo + hi) / 2, txt, ha="right",
                        va="center", fontsize=cfg["fs"], zorder=4)
            else:
                ax.text(COL_X[i] + NODE_W + 0.03, (lo + hi) / 2, txt, ha="left",
                        va="center", fontsize=cfg["fs"], zorder=4,
                        fontweight="bold" if i == 3 else "normal")

    # gebündelte Zielklassen hervorheben
    for t in lvl[2]:
        if t.endswith("_other"):
            lo, hi = pos[2][t]
            ax.add_patch(Rectangle((COL_X[2] - .004, lo - .9), NODE_W + .008,
                                   hi - lo + 1.8, facecolor="none",
                                   edgecolor="#333333", lw=1.1, ls="--", zorder=5))

    # ── Kopfzeilen ──────────────────────────────────────────────────────────
    heads = [f"{len(lvl[0])} diagnostische\nSCP-ECG-Statements",
             f"{len(lvl[1])} diagnostische\nSubklassen (PTB-XL)",
             f"{len(lvl[2])} Zielklassen\n(hybrides Schema)",
             f"{len(lvl[3])} Superklassen\n(Fallback Stufe 3)"]
    top = height + 30
    for x, h in zip(COL_X, heads):
        ax.text(x + NODE_W / 2, top, h, ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.text(COL_X[2] + NODE_W / 2, top - 12,
            "gestrichelt umrandet = gebündelte Restklasse (*_other)",
            ha="center", va="top", fontsize=7.5, style="italic", color="#333333")

    ax.set_xlim(-0.46, COL_X[-1] + 0.66)
    ax.set_ylim(-26, top + 42)
    ax.axis("off")

    fig.suptitle("Labelschema der Stufe-1-Klassifikation — von den SCP-ECG-Statements "
                 "zur Zielklasse", fontsize=13.5, y=0.988)
    ax.text(-0.46, -12,
            f"Fallzahl = Anzahl EKGs mit dem jeweiligen Label im Gesamtdatensatz "
            f"(n = {n_ecg}); Mehrfachlabel je EKG möglich, daher übersteigt die "
            f"Spaltensumme die EKG-Zahl.\n"
            f"Knotenhöhen sind wurzelskaliert (∝ √n) — bei einem Verhältnis von "
            f"9514 (NORM) zu 14 (2AVB) wären seltene Statements linear weder "
            f"sichtbar noch beschriftbar.\n"
            f"Zielklassen zusätzlich mit der Fallzahl im Test-Set (Fold 10), auf "
            f"der die Tragfähigkeitsregel des hybriden Schemas beruht.",
            fontsize=8.5, va="top", color="#444444")

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = OUTFOLDER + "label_schema_sankey.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Gespeichert: {out}")
    print(f"             {out.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
