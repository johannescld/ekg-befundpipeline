# Automatisierte EKG-Befundung

Dreistufiges Verfahren, das aus einem 12-Kanal-Ruhe-EKG einen
strukturierten Befundtext erzeugt, in dem jede Aussage auf einen Messwert
zurückführbar ist.

Entstanden im Rahmen einer Dissertation; entwickelt und intern validiert
auf [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/).

---

## Für den eiligen Leser

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 ekg_config.py          # prüft Pfade, Modelldatei und Rechengerät
python3 ecg_morphology.py      # Selbsttest der Vermessung an sechs EKGs
```

Ein einzelnes EKG befunden:

```bash
python3 ekg_befund.py --file pfad/zum/ekg.hea --hybrid --age 64 --sex m
```

---

## Aufbau

| Stufe | Was passiert | Code |
|---|---|---|
| 1 | Klassifikation des Rohsignals in 19 diagnostische Kategorien (1D-ResNet) | `train_resnet_hybrid.py`, `label_mapping_hybrid.py` |
| 2a | Vermessung aller befundrelevanten Grössen über alle 12 Ableitungen | `ecg_morphology.py` |
| 2b | Zusammenführung zum geprüften Faktum-Set | `build_fact_set_v2.py` |
| 3 | Absicherung: Plausibilität, Kreuzprüfung, Konfidenz-Rückstufung | `plausibility.py`, `build_fact_set_v2.py` |
| 4 | Befundtext (regelbasiert oder per Sprachmodell) | `befund_schema.py`, `llm_befund_v2.py` |

**Stufen 1 bis 3 laufen vollständig lokal und ohne Netzwerkverbindung.**
Nur die Formulierung durch ein Sprachmodell (Stufe 4, `llm_befund_v2.py`)
spricht mit einem externen Dienst; sie ist optional und wird ohne
API-Schlüssel nicht ausgeführt.

---

## Konfiguration

Alle Pfade und die Wahl des Rechengeräts liegen zentral in
`ekg_config.py` und lassen sich über Umgebungsvariablen setzen — im Code
steht kein fester Pfad.

| Variable | Bedeutung | Standard |
|---|---|---|
| `EKG_BASE` | Projektverzeichnis | Verzeichnis von `ekg_config.py` |
| `EKG_DATA` | PTB-XL-Verzeichnis | `$EKG_BASE/ptb-xl-…-1.0.3` |
| `EKG_RESULTS` | Modelle und Ergebnisse | `$EKG_BASE/ptbxl_results` |
| `EKG_DEVICE` | `cuda`, `mps` oder `cpu` erzwingen | automatisch |

```bash
export EKG_BASE=/data/ekg-pipeline
export EKG_DEVICE=cpu
```

Ohne gesetzte Variable wird automatisch CUDA, sonst Apple MPS, sonst CPU
gewählt.

---

## Systemanforderungen

- Python 3.9 oder neuer (entwickelt unter 3.9)
- Betriebssystem beliebig; keine Abhängigkeit von Apple-Hardware
- **Keine Grafikkarte erforderlich.** Die Befundung eines EKGs dauert auf
  einer CPU unter einer Sekunde. Eine GPU lohnt nur für ein erneutes
  Training.
- Arbeitsspeicher: 4 GB genügen für die Anwendung. Für ein Training des
  Modells auf dem vollständigen Datensatz werden etwa 16 GB benötigt.
- Festplatte: der Code belegt wenige MB, PTB-XL etwa 13 GB.

---

## Daten

PTB-XL ist **nicht** Teil dieses Repositories (13 GB). Bezug:

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

Die trainierte Modelldatei (`ptbxl_results/best_resnet1d_hybrid.pt`,
2,8 MB) liegt bei, sodass die Anwendung ohne erneutes Training möglich
ist.

Für die blosse Befundung eigener EKGs wird PTB-XL nicht benötigt — nur
die Modelldatei. Der Datensatz ist nur für Training und Reproduktion der
Auswertungen erforderlich.

### Eigene EKGs einlesen

`ekg_befund.py` akzeptiert WFDB (`.hea`/`.dat`), CSV mit zwölf Spalten
und NumPy-Arrays. Erwartet wird die Ableitungsreihenfolge
I, II, III, aVR, aVL, aVF, V1–V6 in Millivolt. Abweichende Abtastraten
werden automatisch auf 500 Hz umgerechnet, die Länge auf zehn Sekunden
angepasst.

---

## Datenschutz und Betrieb des Sprachmodells

Das Verfahren ist so angelegt, dass es **ohne jede Datenübermittlung nach
aussen** betrieben werden kann.

Klassifikation, Vermessung und Absicherung (Stufen 1 bis 3) sind gewöhnliche
Programme ohne Netzwerkzugriff. Die Formulierung des Befundtextes (Stufe 4)
benötigt ein Sprachmodell, und dafür gibt es zwei Wege:

### Lokales Modell — für Patientendaten

Erwartet wird eine OpenAI-kompatible Schnittstelle, wie sie
[Ollama](https://ollama.com) und [vLLM](https://docs.vllm.ai) bereitstellen.
Die Daten verlassen den Rechner nicht.

```bash
ollama serve &
ollama pull llama3.1:8b

export EKG_LLM_ENDPOINT=http://localhost:11434/v1
export EKG_LLM_MODEL=llama3.1:8b
python3 llm_befund_v2.py
```

Die Formulierungsaufgabe ist eng umrissen: Diagnose, Messwerte und
Satzstruktur sind durch das Faktum-Set vorgegeben, das Modell formuliert
lediglich. Ein Modell der 8- bis 12-Milliarden-Klasse auf einer einzelnen
Grafikkarte mit 24 GB sollte daher genügen; ob das zutrifft, ist Gegenstand
der Untersuchung.

### Externer Dienst — nur für öffentliche Daten

```bash
ANTHROPIC_API_KEY=sk-ant-... python3 llm_befund_v2.py
```

**Für Patientendaten nicht vorgesehen.** Dieser Weg diente der Entwicklung
auf PTB-XL.

### Ohne Konfiguration

Ist weder `EKG_LLM_ENDPOINT` noch `ANTHROPIC_API_KEY` gesetzt, schreibt das
Skript nur die Prompts in eine Datei und erzeugt bewusst keinen Befund,
statt einen Lauf vorzutäuschen.

### Weitere Hinweise

- Das Modell wird bei der Anwendung nicht verändert; es findet kein Lernen
  auf den verarbeiteten Daten statt — weder im Klassifikationsnetz noch im
  Sprachmodell.
- `experiment_pipeline_vs_bild.py` spricht ebenfalls ein Sprachmodell an und
  unterliegt denselben Regeln.

## Reproduktion der Auswertungen

Setzt PTB-XL voraus.

```bash
python3 validate_split.py          # Nachweis der sauberen Datenpartitionierung
python3 predict_test.py            # Vorhersagen für das Test-Set
python3 class_granularity.py       # Gütemasse je Klasse, Konfidenzintervalle
python3 validate_morphology.py     # Validierung der Messgrössen (2.158 EKGs)
python3 plot_label_sankey.py       # Abbildung des Labelschemas
python3 plot_fallbeispiel.py --png # Fallabbildungen
```

Laufzeiten auf einem Notebook ohne GPU: die Validierung der Messgrössen
etwa drei Minuten, die übrigen Schritte jeweils unter einer Minute.
`train_resnet_hybrid.py` trainiert das Modell neu und benötigt mit GPU
etwa eine Stunde.

---

## Stand und bekannte Grenzen

- **QRS-Dauer unzuverlässig.** Zwei unabhängige Messverfahren weichen im
  Median um 57 ms voneinander ab (Maximum 209 ms); das Verfahren über die
  Delineation der Ableitung II liefert bei EKGs ohne Blockbild vereinzelt
  Werte von 180 bis 270 ms. Eine reine Bereichsprüfung kann das nicht
  erkennen, da solche Werte physiologisch möglich sind — nur eben nicht für
  das jeweilige EKG. Die eingebaute Kreuzprüfung (`pruefe_qrs_dauer` in
  `build_fact_set_v2.py`) stellt daher beide Messwege gegeneinander und
  verwirft bei Widerspruch den Wert, statt eine falsche Zahl in den Befund zu
  schreiben. In einer Stichprobe von 20 EKGs betraf das 13 Fälle. Eine
  belastbare QRS-Bestimmung ist der derzeit dringlichste offene Punkt.
- **Grenzwerte noch ohne Literaturbeleg.** Die Schwellen für Pardée-Q,
  ST-Hebung und Lagetyp sind nach gängiger klinischer Konvention gesetzt
  und im Code als belegpflichtig markiert.
- **Befundqualität nicht bewertet.** Die Validierung prüft die
  Messgrössen, nicht den formulierten Text. Eine kardiologische Bewertung
  steht aus.

---

## Mitgelieferte Belege

| Datei | Inhalt |
|---|---|
| `ptbxl_results/fallbeispiele.pdf` | sechs Fälle: EKG, eingezeichnete Messpunkte, Befund |
| `ptbxl_results/label_schema_sankey.pdf` | Labelschema: 44 SCP-Statements bis 5 Superklassen |
| `ptbxl_results/granularity_analysis.pdf` | Konfidenzintervalle je Klasse, Sensitivitätsanalyse |
| `ptbxl_results/class_metrics_hybrid.csv` | AUROC, Fmax, Average Precision je Zielklasse |
| `split_validation_report.json` | Nachweis der sauberen Datenpartitionierung |
| `ptbxl_results/morphology_validation.json` | Validierung der Messgrössen gegen 2.158 EKGs |
| `befund_prompts_v2.txt` | System- und User-Prompt im Wortlaut |
| `example_fact_sets_v2.json` | Beispiel-Faktum-Sets als Eingang für Stufe 4 |

## Lizenz

MIT-Lizenz, siehe `LICENSE`. Nutzung, Veränderung und Weitergabe sind
gestattet, solange die Urheberangabe erhalten bleibt. Eine Gewährleistung
besteht nicht.

**Dies ist Forschungscode und kein Medizinprodukt.** Er ist nicht für die
Anwendung am Patienten bestimmt. Erzeugte Befunde ersetzen keine ärztliche
Beurteilung.

## Datengrundlage und Zitation

Entwickelt auf PTB-XL:

> Wagner, P., Strodthoff, N., Bousseljot, RD. et al. PTB-XL, a large publicly
> available electrocardiography dataset. *Sci Data* 7, 154 (2020).
> https://doi.org/10.1038/s41597-020-0495-6

PTB-XL steht unter Creative Commons Attribution 4.0. Die beiliegende
Modelldatei wurde auf diesem Datensatz trainiert.

Die Architektur des Klassifikationsnetzes folgt der Benchmark-Implementierung:

> Strodthoff, N., Wagner, P., Schaeffter, T., Samek, W. Deep Learning for ECG
> Analysis: Benchmarks and Insights from PTB-XL. *IEEE J Biomed Health Inform*
> 25(5), 1519–1528 (2021).
