"""
Experiment: strukturierte Pipeline gegen direkte Bildbefundung.

Fragestellung
-------------
Rechtfertigt der Aufwand der Pipeline (Klassifikation -> Messung -> geprüftes
Faktum-Set -> Formulierung) den Ertrag, verglichen mit dem naheliegenden
Vorgehen, einem multimodalen Sprachmodell schlicht das EKG-Bild zu zeigen?

Zwei Arme, dieselben EKGs, dasselbe Sprachmodell:

  Arm A "Pipeline"   Eingang ist das geprüfte Faktum-Set (Messwerte aus dem
                     Rohsignal, sprachlich eingestufte Verdachtsdiagnose).
                     Das Modell formuliert nur.

  Arm B "Bild"       Eingang ist das gerenderte 12-Kanal-EKG als Bild plus
                     Alter und Geschlecht — mehr nicht. Das Modell muss
                     selbst ablesen, messen und beurteilen.

Beide Arme erhalten dieselbe Befundstruktur vorgegeben, damit die Texte
vergleichbar sind und der Unterschied nicht bloss am Format hängt.

Verblindung: Arm B sieht weder Faktum-Set noch Klassifikationsergebnis noch
die Referenzdiagnose. Arm A sieht die Referenzdiagnose ebenfalls nicht — sie
steht nur im Auswertungsprotokoll.

Ohne API-Schlüssel werden Bilder und Prompts erzeugt und abgelegt, aber kein
Modelllauf ausgeführt.

Aufruf:
    python3 experiment_pipeline_vs_bild.py --n 60 --seed 42
    ANTHROPIC_API_KEY=sk-ant-... python3 experiment_pipeline_vs_bild.py --n 60
"""

import argparse
import ast
import base64
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from ekg_config import DATAFOLDER, OUTFOLDER, RESULTS
from label_mapping_hybrid import build_label_matrix, TARGET_TO_SUPER
from befund_schema import SYSTEM_PROMPT, BEFUND_SCHEMA, build_user_prompt

SR = 500
MODEL = "claude-opus-4-5"      # beim Lauf auf die eingesetzte Version setzen
EXP_DIR = RESULTS / "experiment_bild_vs_pipeline"
BILD_DIR = EXP_DIR / "ekg_bilder"

# Arm B erhält dieselbe Befundstruktur, aber keine Messwerte.
SYSTEM_PROMPT_BILD = """Du bist ein erfahrener Kardiologe und formulierst \
EKG-Befunde für die klinische Dokumentation.

Du erhältst ein 12-Kanal-Ruhe-EKG als Abbildung (25 mm/s, 10 mm/mV) sowie \
Alter und Geschlecht. Beurteile es eigenständig.

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

REGELN:
- Gib Messwerte mit Zahl und Einheit an (z. B. "QRS 110 ms").
- Was du nicht sicher bestimmen kannst, benenne ausdrücklich als nicht \
beurteilbar. Rate nicht.
- Negativbefunde gehören in den Text ("Kein Pardée-Q.", "Keine Extrasystolen.").
- Sprache: medizinisches Deutsch, knapp.
- Antworte NUR mit dem JSON-Objekt nach dem vorgegebenen Schema."""


def render_ekg_bild(sig: np.ndarray, pfad: Path) -> None:
    """12-Kanal-EKG im klinischen 3x4-Layout als PNG, wie ein Ausdruck."""
    from plot_fallbeispiel import plot_ekg_12
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(11.7, 8.3), dpi=150)
    gs = GridSpec(4, 1, figure=fig, left=0.02, right=0.99,
                  top=0.98, bottom=0.02, hspace=0.06)
    plot_ekg_12(fig, gs, sig)
    fig.savefig(pfad, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def stichprobe(Y, y, classes, n, seed):
    """
    Stratifizierte Stichprobe über die fünf Superklassen aus Fold 10.

    Gleichverteilung über die Superklassen statt zufälliger Ziehung: Bei
    natürlicher Prävalenz bestünde die Stichprobe zur Hälfte aus
    Normalbefunden, an denen sich Faktentreue kaum unterscheiden lässt.
    """
    rng = np.random.default_rng(seed)
    lab = pd.DataFrame(y, index=Y.index, columns=classes)
    supers = ["NORM", "MI", "STTC", "CD", "HYP"]
    je = max(1, n // len(supers))
    gewaehlt = []
    for sup in supers:
        spalten = [c for c in classes if TARGET_TO_SUPER[c] == sup]
        maske = lab[spalten].sum(axis=1) > 0
        kandidaten = lab.index[maske].difference(gewaehlt)
        if len(kandidaten) == 0:
            continue
        gewaehlt += list(rng.choice(kandidaten,
                                    size=min(je, len(kandidaten)),
                                    replace=False))
    return sorted(int(e) for e in gewaehlt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="Stichprobengrösse")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--factsets", default=None,
                    help="vorhandene Faktum-Sets (sonst werden sie erzeugt)")
    args = ap.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    BILD_DIR.mkdir(parents=True, exist_ok=True)

    Y = pd.read_csv(DATAFOLDER + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    mlb, y_full = build_label_matrix(Y)
    classes = list(mlb.classes_)
    maske = y_full.sum(axis=1) > 0
    Y_f, y_f = Y[maske], y_full[maske]
    test = Y_f.strat_fold == 10
    ids = stichprobe(Y_f[test], y_f[test.values], classes, args.n, args.seed)
    print(f"Stichprobe: {len(ids)} EKGs aus Fold 10 (Seed {args.seed})")

    # ── Faktum-Sets für Arm A ───────────────────────────────────────────────
    fs_datei = args.factsets or str(EXP_DIR / "fact_sets.json")
    if not Path(fs_datei).exists():
        print("Erzeuge Faktum-Sets (Stufe 1 + Messung + Prüfung) …")
        os.system(f'python3 build_fact_set_v2.py --ids {" ".join(map(str, ids))} '
                  f'--out "{fs_datei}" > /dev/null 2>&1')
    with open(fs_datei, encoding="utf-8") as f:
        fact_sets = {fs["ecg_id"]: fs for fs in json.load(f)}

    # ── Bilder für Arm B ────────────────────────────────────────────────────
    X = np.load(DATAFOLDER + "raw500_signals.npy", mmap_mode="r")
    pos = {e: i for i, e in enumerate(Y.index)}
    print("Rendere EKG-Bilder …")
    for eid in ids:
        p = BILD_DIR / f"ekg_{eid}.png"
        if not p.exists():
            render_ekg_bild(np.asarray(X[pos[eid]], dtype=float), p)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Prompts dokumentieren, keinen Lauf vortäuschen
        with open(EXP_DIR / "prompts.txt", "w", encoding="utf-8") as f:
            f.write("ARM A — SYSTEM\n" + "=" * 70 + "\n" + SYSTEM_PROMPT)
            f.write("\n\nARM B — SYSTEM\n" + "=" * 70 + "\n" + SYSTEM_PROMPT_BILD)
            f.write("\n\nSCHEMA (beide Arme)\n" + "=" * 70 + "\n")
            f.write(json.dumps(BEFUND_SCHEMA, indent=2, ensure_ascii=False))
            beispiel = next(iter(fact_sets.values()))
            f.write("\n\nARM A — USER (Beispiel)\n" + "=" * 70 + "\n")
            f.write(build_user_prompt(beispiel))
            f.write("\n\nARM B — USER (Beispiel)\n" + "=" * 70 + "\n")
            f.write("[EKG-Bild]\nPatient: 64 Jahre, männlich\n"
                    "Erstelle den strukturierten EKG-Befund als JSON.")
        with open(EXP_DIR / "stichprobe.json", "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "seed": args.seed}, f, indent=2)
        print(f"\nVorbereitet in {EXP_DIR}")
        print(f"  {len(ids)} EKG-Bilder, Faktum-Sets, Prompts, Stichprobenliste")
        print("\nKein ANTHROPIC_API_KEY gesetzt — kein Modelllauf ausgeführt.")
        print("Zum Durchführen:")
        print("  ANTHROPIC_API_KEY=sk-ant-... python3 experiment_pipeline_vs_bild.py")
        return

    # ── Modelllauf beider Arme ──────────────────────────────────────────────
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    ergebnisse = []

    for i, eid in enumerate(ids, 1):
        fs = fact_sets.get(eid)
        if fs is None:
            continue
        print(f"[{i}/{len(ids)}] EKG {eid}")

        # Arm A
        rA = client.messages.create(
            model=MODEL, max_tokens=1500, temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(fs)}])

        # Arm B
        bild = base64.standard_b64encode(
            (BILD_DIR / f"ekg_{eid}.png").read_bytes()).decode()
        pat = fs["patient"]
        rB = client.messages.create(
            model=MODEL, max_tokens=1500, temperature=0.2,
            system=SYSTEM_PROMPT_BILD,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png",
                                             "data": bild}},
                {"type": "text",
                 "text": f"Patient: {pat['age_years']} Jahre, {pat['sex']}\n"
                         f"Erstelle den strukturierten EKG-Befund als JSON."}]}])

        def parse(resp):
            roh = resp.content[0].text
            m = re.search(r"\{.*\}", roh, re.DOTALL)
            try:
                return json.loads(m.group(0) if m else roh), roh
            except json.JSONDecodeError:
                return None, roh

        bA, rohA = parse(rA)
        bB, rohB = parse(rB)
        ergebnisse.append({
            "ecg_id": eid, "modell": MODEL,
            "true_classes": fs.get("true_classes"),
            "arm_a_pipeline": {"befund": bA, "roh": rohA,
                               "tokens_in": rA.usage.input_tokens,
                               "tokens_out": rA.usage.output_tokens},
            "arm_b_bild": {"befund": bB, "roh": rohB,
                           "tokens_in": rB.usage.input_tokens,
                           "tokens_out": rB.usage.output_tokens},
        })

        with open(EXP_DIR / "ergebnisse.json", "w", encoding="utf-8") as f:
            json.dump(ergebnisse, f, indent=2, ensure_ascii=False)

    print(f"\nFertig: {EXP_DIR / 'ergebnisse.json'}")
    print("Auswertung:  python3 evaluate_faktentreue.py")


if __name__ == "__main__":
    main()
