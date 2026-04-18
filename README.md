# DataPreprocessor

Kleine, klare Pipeline zur Aufbereitung von de-en Beispielen fuer Seq2Seq-Training:
`load -> norm -> filter -> tokenize`.

## Hauptfunktionen
- `load`: laedt Rohdaten konsistent und reproduzierbar.
- `norm`: normalisiert Texte (z. B. Whitespace/Zeichenbereinigung) ohne Seiteneffekte am Input.
- `filter`: entfernt problematische Beispiele ueber klar definierte Praedikate.
- `tokenize`: erzeugt modellfertige Token-Features fuer beide Sprachen.
  Wenn `max_seq_len` konfiguriert ist, verwirft `tokenize` ausserdem komplette
  Beispiele, deren untrunkierte `src`- oder `tgt`-Tokenliste diese Grenze
  ueberschreitet.

## Mehrwert gegenueber ad-hoc Implementierungen
- Einfache End-to-End-Orchestrierung statt Boilerplate pro Experiment.
- Reporting ist optional integrierbar, ohne die Kernlogik zu verkomplizieren.
- Saubere Trennung zwischen Kern-Transformationen und I/O/Runner-Code.
- Regeln sind klar ausgelagert (`norm/changes.py`, `filter/predicates/*`): praezise, kompakt und leicht anpassbar/erweiterbar.
- Zwischenergebnisse sind pro Stage als JSONL leicht inspizierbar: grober Check schnell, Details jederzeit nachvollziehbar.
- Visualisierung kann direkt auf Reports aufbauen (bereits begonnen) und schrittweise wachsen, ohne den Pipeline-Kern umzubauen.
- Fuer Translation-Training materialisiert die Pipeline einen expliziten Ziel-BOS-Token, falls der verwendete Tokenizer
  keinen liefert (z. B. Marian/OPUS-MT). Das vermeidet eine schwer erkennbare Inkompatibilitaet zwischen Tokenizer-Ausgabe
  und Seq2Seq-Training.

## Gold, Silber, Bronze (Zielbild)
Aktuell sind diese Staerken als Keime angelegt. Ziel ist, sie in voller Praxistiefe auszubauen:
- Gold: klare, reproduzierbare End-to-End-Pipeline statt ad-hoc Skriptverkettung.
- Silber: konsequente Trennung von Fachlogik und I/O/Runner fuer bessere Testbarkeit und Wartbarkeit.
- Bronze: transparente Reports/JSONL plus wachsende Visualisierung fuer schnelle Uebersicht und Detail-Drilldown.

## Coming Soon
- Reproduzierbare Pipeline-Runs ueber zentrale Konfiguration (YAML/JSON).
- Einheitliche Stage-Statistiken und Run-Metadaten (Counts, Parameter, Artefakte).
- Strukturiertere Datentypen fuer `Example` (z. B. `TypedDict`) fuer bessere Safety.
- Standardisierte Report-Formate (JSONL) als robuste Basis fuer Visualisierung.
- Weitere Visualisierungen fuer Stage-Zwischenstaende: schnelle grafische Grobuebersicht plus Detail-Drilldown in JSONL.
- Klarer Zielzustand: deklarative, nachvollziehbare, idempotente Datenpipeline vom Rohdatensatz bis zum trainingsfertigen Output.

## Voraussetzungen
- Python 3.13 (oder kompatibel)
- `datasets`
- `transformers`
- `sentencepiece`
- `sacremoses` (wichtig bei Marian/OPUS-MT-Tokenizern wie `Helsinki-NLP/opus-mt-de-en`)

Beispiel:
`python -m pip install datasets transformers sentencepiece sacremoses`
