"""
Zentrale Konfiguration — Pfade und Rechengerät.

Ersetzt die zuvor fest einkodierten Pfade, damit die Pipeline auf einem
anderen Rechner (Klinik-Arbeitsplatz, Server, VPN-Zugang) unverändert läuft.

Auflösung der Verzeichnisse, in dieser Reihenfolge:
  1. Umgebungsvariable, falls gesetzt
  2. Verzeichnis dieser Datei (Standardfall: alles liegt beieinander)

Umgebungsvariablen
------------------
  EKG_BASE       Projektverzeichnis (Standard: Verzeichnis dieser Datei)
  EKG_DATA       PTB-XL-Verzeichnis (nur für Training und Auswertung nötig)
  EKG_RESULTS    Verzeichnis für Modelle und Ergebnisse
  EKG_DEVICE     Rechengerät erzwingen: cuda | mps | cpu

Beispiel (Linux/macOS):
    export EKG_BASE=/data/ekg-pipeline
    export EKG_DEVICE=cpu

Beispiel (Windows, PowerShell):
    $env:EKG_BASE = "D:\\ekg-pipeline"
    $env:EKG_DEVICE = "cpu"
"""

import os
from pathlib import Path


# ─── Verzeichnisse ───────────────────────────────────────────────────────────
BASE = Path(os.environ.get("EKG_BASE", Path(__file__).resolve().parent))

PTBXL_DIRNAME = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

DATA = Path(os.environ.get("EKG_DATA", BASE / PTBXL_DIRNAME))
RESULTS = Path(os.environ.get("EKG_RESULTS", BASE / "ptbxl_results"))

# Als String mit abschliessendem Trennzeichen — die bestehenden Skripte
# setzen Pfade per Konkatenation zusammen (DATAFOLDER + "ptbxl_database.csv").
BASE_STR = str(BASE) + os.sep
DATAFOLDER = str(DATA) + os.sep
OUTFOLDER = str(RESULTS) + os.sep

RESULTS.mkdir(parents=True, exist_ok=True)


# ─── Rechengerät ─────────────────────────────────────────────────────────────
def get_device(verbose: bool = False) -> str:
    """
    Wählt das Rechengerät: CUDA (Klinik-Server, Workstation mit NVIDIA-Karte),
    sonst MPS (Apple Silicon), sonst CPU.

    Die Pipeline läuft vollständig auf der CPU; für die Befundung eines
    einzelnen EKGs genügt das (Inferenz im Sekundenbereich). Nur ein
    erneutes Training profitiert spürbar von einer GPU.
    """
    forced = os.environ.get("EKG_DEVICE")
    if forced:
        if verbose:
            print(f"Rechengerät (erzwungen): {forced}")
        return forced

    import torch
    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    if verbose:
        print(f"Rechengerät: {dev}")
    return dev


def check_environment(require_data: bool = False) -> None:
    """Prüft die Umgebung und meldet fehlende Bestandteile verständlich."""
    probleme = []

    if not RESULTS.exists():
        probleme.append(f"Ergebnisverzeichnis fehlt: {RESULTS}")

    modell = RESULTS / "best_resnet1d_hybrid.pt"
    if not modell.exists():
        probleme.append(
            f"Modelldatei fehlt: {modell}\n"
            f"      Sie gehört zum Lieferumfang und muss mitkopiert werden.")

    if require_data:
        db = DATA / "ptbxl_database.csv"
        if not db.exists():
            probleme.append(
                f"PTB-XL-Daten fehlen: {db}\n"
                f"      Nur für Training und Auswertung nötig, nicht für die "
                f"Befundung einzelner EKGs.")

    try:
        import torch, neurokit2, wfdb  # noqa: F401
    except ImportError as e:
        probleme.append(f"Python-Paket fehlt: {e.name}  "
                        f"(siehe requirements.txt)")

    if probleme:
        print("Die Umgebung ist nicht vollständig:\n")
        for p in probleme:
            print(f"  - {p}")
        print(f"\nProjektverzeichnis: {BASE}")
        print("Anpassen über die Umgebungsvariablen EKG_BASE / EKG_DATA / "
              "EKG_RESULTS.")
        raise SystemExit(1)


if __name__ == "__main__":
    print(f"Projektverzeichnis : {BASE}")
    print(f"PTB-XL-Daten       : {DATA}    "
          f"{'(vorhanden)' if (DATA / 'ptbxl_database.csv').exists() else '(fehlt)'}")
    print(f"Ergebnisse/Modelle : {RESULTS}")
    modell = RESULTS / "best_resnet1d_hybrid.pt"
    print(f"Modelldatei        : "
          f"{'vorhanden' if modell.exists() else 'FEHLT'}  ({modell.name})")
    print(f"Rechengerät        : {get_device()}")
