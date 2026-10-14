"""
Granularitätsanalyse: Warum 19 Zielklassen?
(Betreuer-Frage: "Sind die gewählten 19 Zielklassen das Optimum und wenn ja wieso?")

Die Wahl der Granularität ist ein Zielkonflikt:
  feiner  -> klinisch informativere Befunde
  feiner  -> weniger Positivfälle pro Klasse -> unzuverlässigere Metrik-Schätzung

Dieses Skript quantifiziert die zweite Seite empirisch:

  A) Bootstrap-95%-Konfidenzintervalle der AUROC pro Zielklasse
     -> zeigt, wie die Schätzunsicherheit mit sinkender Fallzahl explodiert
  B) Zusammenhang KI-Breite ~ Fallzahl (Kurvenfit), daraus die Ableitung
     einer Mindest-Fallzahl für eine belastbare Klassenmetrik
  C) Sensitivitätsanalyse: Zahl der Zielklassen bei n_min ∈ {10,20,30,50,100}
  D) Zusätzliche Metriken neben AUROC: Fmax und Average Precision (AP)
     -> AUROC überschätzt seltene Klassen systematisch (Prävalenzeffekt)

Aufruf:  python3 class_granularity.py
Ausgabe: ptbxl_results/class_metrics_hybrid.csv
         ptbxl_results/granularity_analysis.png
"""

import ast
import json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from label_mapping_hybrid import (SUBCLASS_TO_TARGET, TARGET_TO_SUPER,

                                  TARGET_LABELS_DE, _build_code_to_target)

from ekg_config import BASE_STR  # zentrale Pfadkonfiguration

DATAFOLDER = BASE_STR + "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTFOLDER = BASE_STR + "ptbxl_results/"

N_BOOT = 2000
RNG = np.random.default_rng(42)


def fmax(y_true, y_score):
    """Maximaler F1-Score über alle Schwellenwerte (Strodthoff et al.)."""
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    return float(f1.max())


def bootstrap_auroc(y_true, y_score, n_boot=N_BOOT):
    """Stratifiziertes Bootstrap-KI der AUROC (resampled über EKGs)."""
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        vals.append(roc_auc_score(yt, y_score[idx]))
    if not vals:
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), float(np.std(vals))


def main():
    d = np.load(OUTFOLDER + "test_predictions_hybrid.npz", allow_pickle=True)
    preds, true = d["preds"], d["true"]
    class_names = [str(c) for c in d["class_names"]]

    # ── A/D: Metriken + Bootstrap-KI pro Klasse ─────────────────────────────
    print("=" * 92)
    print("  Metriken pro Zielklasse — Test (Fold 10), Bootstrap-95%-KI aus "
          f"{N_BOOT} Resamples")
    print("=" * 92)
    print(f"{'Klasse':<11} {'Super':<5} {'n':>5} {'Präv.':>6} {'AUROC':>7} "
          f"{'95%-KI':>16} {'Breite':>7} {'Fmax':>6} {'AP':>6}")
    print("-" * 92)

    rows = []
    for i, cls in enumerate(class_names):
        yt, ys = true[:, i], preds[:, i]
        n_pos = int(yt.sum())
        auc = roc_auc_score(yt, ys)
        lo, hi, sd = bootstrap_auroc(yt, ys)
        rows.append({
            "Klasse": cls,
            "Superklasse": TARGET_TO_SUPER[cls],
            "Bezeichnung": TARGET_LABELS_DE[cls],
            "n_test_pos": n_pos,
            "praevalenz": n_pos / len(yt),
            "AUROC": auc,
            "AUROC_KI_lo": lo,
            "AUROC_KI_hi": hi,
            "AUROC_KI_breite": hi - lo,
            "AUROC_bootstrap_sd": sd,
            "Fmax": fmax(yt, ys),
            "AP": average_precision_score(yt, ys),
        })

    df = pd.DataFrame(rows).sort_values("n_test_pos", ascending=False)
    for _, r in df.iterrows():
        print(f"{r.Klasse:<11} {r.Superklasse:<5} {r.n_test_pos:>5} "
              f"{r.praevalenz:>6.1%} {r.AUROC:>7.3f} "
              f"[{r.AUROC_KI_lo:.3f}–{r.AUROC_KI_hi:.3f}] "
              f"{r.AUROC_KI_breite:>7.3f} {r.Fmax:>6.3f} {r.AP:>6.3f}")

    print("-" * 92)
    print(f"{'Makro':<11} {'':<5} {int(df.n_test_pos.sum()):>5} {'':>6} "
          f"{df.AUROC.mean():>7.3f} {'':>16} {df.AUROC_KI_breite.mean():>7.3f} "
          f"{df.Fmax.mean():>6.3f} {df.AP.mean():>6.3f}")

    # ── B: KI-Breite als Funktion der Fallzahl ──────────────────────────────
    print("\n" + "=" * 92)
    print("  [B] Schätzunsicherheit vs. Fallzahl")
    print("=" * 92)
    n = df.n_test_pos.values.astype(float)
    w = df.AUROC_KI_breite.values
    # Erwartung: Breite ~ c / sqrt(n)
    c = float(np.median(w * np.sqrt(n)))
    print(f"    Fit  Breite ≈ {c:.2f} / √n   (Median-Schätzer)")
    for nn in [10, 20, 30, 50, 100, 200, 500]:
        print(f"      n = {nn:>3}  ->  erwartete KI-Breite ≈ {c/np.sqrt(nn):.3f} "
              f"AUROC-Punkte")

    print("\n    Beobachtete Breiten (die Fallzahl ist nur ein Teil der Erklärung —")
    print("    schwach trennbare Klassen streuen zusätzlich):")
    for _, r in df.nlargest(5, "AUROC_KI_breite").iterrows():
        print(f"      {r.Klasse:<10} n={int(r.n_test_pos):>4}  Breite {r.AUROC_KI_breite:.3f}"
              f"  (Fit erwartet {c/np.sqrt(r.n_test_pos):.3f})")

    delta_rf = 0.9272 - 0.7951      # ResNet vs. RF-Baseline, Makro
    n_crit = (c / delta_rf) ** 2
    print(f"\n    Interpretation: Der Makro-Abstand zur RF-Baseline beträgt "
          f"Δ = {delta_rf:.3f}.")
    print(f"    Nach dem Fit wird das 95%-KI erst unterhalb von n ≈ {n_crit:.0f} "
          f"breiter als")
    print("    dieser Abstand. Die Fallzahl allein rechtfertigt eine Schwelle von 20")
    print("    also NICHT — entscheidend ist die beobachtete Streuung: die drei")
    print("    kleinsten Klassen (CD_other 16, MI_other 22, HYP_other 23) liegen")
    print("    mit Breiten von 0.078-0.290 deutlich über dem Fit, CD_other mit")
    print("    [0.693-0.983] ist faktisch nicht interpretierbar.")

    # ── C: Sensitivitätsanalyse der Mindestfallzahl ─────────────────────────
    print("\n" + "=" * 92)
    print("  [C] Sensitivitätsanalyse: Zielklassenzahl in Abhängigkeit von n_min")
    print("=" * 92)

    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    code_to_sub = {}
    scp = pd.read_csv(DATAFOLDER + "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1.0]
    for code, row in diag.iterrows():
        if pd.notna(row.diagnostic_subclass):
            code_to_sub[code] = row.diagnostic_subclass

    test = Y[Y.strat_fold == 10]
    sub_counts = Counter()
    for codes in test.scp_codes:
        for s in {code_to_sub[c] for c in codes if c in code_to_sub}:
            sub_counts[s] += 1

    sub_to_super = {s: TARGET_TO_SUPER[t] for s, t in SUBCLASS_TO_TARGET.items()}

    print("Regel: Subklasse eigenständig, wenn n_test >= n_min; Rest je")
    print("Superklasse zu '<SUPER>_other' gebündelt (Bündel wird nur geführt,")
    print("wenn es selbst n_min erreicht).\n")
    print(f"{'n_min':>6} {'eigenst. Klassen':>17} {'+ *_other-Bündel':>17} "
          f"{'= Zielklassen':>14} {'min. n':>7}")
    print("-" * 92)
    sens = []
    for n_min in [1, 10, 20, 30, 50, 100, 200]:
        keep = [s for s in SUBCLASS_TO_TARGET if sub_counts.get(s, 0) >= n_min]
        drop = [s for s in SUBCLASS_TO_TARGET if s not in keep]
        buckets = sorted({sub_to_super[s] for s in drop})
        bucket_n = {b: sum(sub_counts.get(s, 0) for s in drop if sub_to_super[s] == b)
                    for b in buckets}
        valid_buckets = [b for b, v in bucket_n.items() if v >= n_min]
        total = len(keep) + len(valid_buckets)
        min_n = min([sub_counts.get(s, 0) for s in keep] +
                    [bucket_n[b] for b in valid_buckets]) if total else 0
        print(f"{n_min:>6} {len(keep):>17} {len(valid_buckets):>17} "
              f"{total:>14} {min_n:>7}")
        sens.append({"n_min": n_min, "eigenstaendig": len(keep),
                     "buendel": len(valid_buckets), "zielklassen": total,
                     "kleinste_klasse": min_n})

    # ── Abgleich: implementiertes Schema vs. strikte Regel bei n_min = 20 ────
    print("\n" + "-" * 92)
    print("  ABGLEICH implementiertes Schema (19 Klassen) vs. strikte Regel n_min = 20")
    print("-" * 92)
    n_min = 20
    strict_keep = {s for s in SUBCLASS_TO_TARGET if sub_counts.get(s, 0) >= n_min}
    impl_keep = {s for s, t in SUBCLASS_TO_TARGET.items() if not t.endswith("_other")}

    dev_bundled = sorted(strict_keep - impl_keep)      # Regel: eigenständig, real: gebündelt
    impl_buckets = sorted({t for t in SUBCLASS_TO_TARGET.values() if t.endswith("_other")})
    bucket_counts = {b: sum(sub_counts.get(s, 0)
                            for s, t in SUBCLASS_TO_TARGET.items() if t == b)
                     for b in impl_buckets}
    dev_buckets = {b: v for b, v in bucket_counts.items() if v < n_min}

    print(f"    Implementiert: {len(impl_keep)} eigenständige + "
          f"{len(impl_buckets)} Bündel = {len(impl_keep) + len(impl_buckets)} Zielklassen")
    print(f"    Strikte Regel: {len(strict_keep)} eigenständige + gültige Bündel "
          f"= {sens[2]['zielklassen']} Zielklassen")
    print("\n    Abweichung 1 — trotz erfüllter Schwelle gebündelt:")
    for s in dev_bundled:
        print(f"      {s:<10} n_test = {sub_counts.get(s,0)}  ->  "
              f"{SUBCLASS_TO_TARGET[s]}   (Regel hätte eigene Klasse ergeben)")
    if not dev_bundled:
        print("      keine")
    print("\n    Abweichung 2 — Bündel unterhalb der Schwelle trotzdem geführt:")
    for b, v in dev_buckets.items():
        members = [s for s, t in SUBCLASS_TO_TARGET.items() if t == b]
        print(f"      {b:<10} n_test = {v}  (= {' + '.join(members)})")
    if not dev_buckets:
        print("      keine")

    print("\n    -> Das Schema ist damit KEINE strikte Anwendung der 20er-Regel,")
    print("       sondern eine Designentscheidung mit zwei begründungspflichtigen")
    print("       Abweichungen. Für die Arbeit muss entweder die Regel als")
    print("       Richtwert deklariert und beide Abweichungen klinisch begründet")
    print("       werden, oder das Schema wird an die Regel angeglichen.")
    print("       Empirisch (s. [A]) ist CD_other mit KI [0.693-0.983] die")
    print("       schwächste Klasse des Schemas.")

    report_dev = {
        "implementiert_zielklassen": len(impl_keep) + len(impl_buckets),
        "strikte_regel_zielklassen": sens[2]["zielklassen"],
        "abweichung_gebuendelt_trotz_schwelle":
            {s: sub_counts.get(s, 0) for s in dev_bundled},
        "abweichung_buendel_unter_schwelle": dev_buckets,
    }

    # ── Speichern ───────────────────────────────────────────────────────────
    out_csv = OUTFOLDER + "class_metrics_hybrid.csv"
    df.to_csv(out_csv, index=False)
    with open(OUTFOLDER + "granularity_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump({"fit_c": c, "sensitivity": sens,
                   "schema_abgleich": report_dev}, f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert: {out_csv}")

    # ── Abbildung ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    colors = {"NORM": "#4C72B0", "MI": "#C44E52", "STTC": "#DD8452",
              "CD": "#55A868", "HYP": "#8172B3"}
    for _, r in df.iterrows():
        ax.errorbar(r.n_test_pos, r.AUROC,
                    yerr=[[r.AUROC - r.AUROC_KI_lo], [r.AUROC_KI_hi - r.AUROC]],
                    fmt="o", ms=5, capsize=3, lw=1.2,
                    color=colors[r.Superklasse], alpha=.85)
        if r.n_test_pos < 60 or r.AUROC < 0.87:
            off = (4, -11) if r.Klasse in ("CD_other", "LAO/LAE", "IVCD") else (5, 4)
            ax.annotate(r.Klasse, (r.n_test_pos, r.AUROC), fontsize=7,
                        xytext=off, textcoords="offset points")
    ax.axvline(20, color="k", ls="--", lw=1)
    ax.text(22, 0.695, "n = 20\n(Schwelle laut Doku)", fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Positivfälle im Test-Set (log)")
    ax.set_ylabel("AUROC mit 95%-Bootstrap-KI")
    ax.set_title("A  Schätzunsicherheit pro Zielklasse")
    ax.grid(alpha=.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=s)
               for s, c in colors.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower right", title="Superklasse",
              title_fontsize=7)

    ax = axes[1]
    ax.scatter(n, w, s=28, color="#4C72B0", zorder=3)
    xs = np.linspace(10, n.max(), 200)
    ax.plot(xs, c / np.sqrt(xs), "r-", lw=1.5, label=f"$\\approx {c:.2f}/\\sqrt{{n}}$")
    ax.axhline(0.132, color="gray", ls=":", lw=1.2)
    ax.text(n.max() * .35, .142, "Δ ResNet vs. RF-Baseline (0.132)",
            fontsize=7.5, color="gray")
    ax.axvline(20, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Positivfälle im Test-Set (log)")
    ax.set_ylabel("Breite des 95%-KI (AUROC-Punkte)")
    ax.set_title("B  Unsicherheit ~ Fallzahl")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[2]
    s = pd.DataFrame(sens)
    ax.plot(s.n_min, s.zielklassen, "o-", color="#55A868", lw=1.8,
            label="strikte Regel")
    for _, r in s.iterrows():
        off = (-13, -3) if r.n_min == 20 else (0, 7)
        ax.annotate(int(r.zielklassen), (r.n_min, r.zielklassen), fontsize=8,
                    xytext=off, textcoords="offset points", ha="center")
    ax.axvline(20, color="k", ls="--", lw=1)
    ax.plot([20], [report_dev["implementiert_zielklassen"]], "D", ms=9,
            color="#C44E52", zorder=5, label="implementiertes Schema")
    ax.annotate(f"{report_dev['implementiert_zielklassen']}\n(2 Abweichungen)",
                (20, report_dev["implementiert_zielklassen"]), fontsize=7.5,
                color="#C44E52", xytext=(9, -4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Mindest-Fallzahl $n_{min}$ im Test-Set (log)")
    ax.set_ylabel("resultierende Zielklassen")
    ax.set_title("C  Granularität vs. Tragfähigkeit")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=.3)

    fig.suptitle("Begründung der 19 Zielklassen: klinische Auflösung vs. "
                 "statistische Belastbarkeit", fontsize=11)
    fig.tight_layout()
    out_png = OUTFOLDER + "granularity_analysis.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Abbildung:   {out_png}")


if __name__ == "__main__":
    main()
