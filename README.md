# DataPreprocessor

Kleine, klare Pipeline zur Aufbereitung von de-en Beispielen fuer Seq2Seq-Training:
`download -> norm -> filter -> tokenize`.

## Hauptfunktionen
- `download`: laedt Rohdaten konsistent und reproduzierbar.
- `norm`: normalisiert Texte (z. B. Whitespace/Zeichenbereinigung) ohne Seiteneffekte am Input.
- `filter`: entfernt problematische Beispiele ueber klar definierte Praedikate.
- `tokenize`: erzeugt modellfertige Token-Features fuer beide Sprachen.

## Mehrwert gegenueber ad-hoc Implementierungen
- Einfache End-to-End-Orchestrierung statt Boilerplate pro Experiment.
- Reporting ist optional integrierbar, ohne die Kernlogik zu verkomplizieren.
- Saubere Trennung zwischen Kern-Transformationen und I/O/Runner-Code.
- Regeln sind klar ausgelagert (`norm/changes.py`, `filter/predicates/*`): praezise, kompakt und leicht anpassbar/erweiterbar.
- Zwischenergebnisse sind pro Stage als JSONL leicht inspizierbar: grober Check schnell, Details jederzeit nachvollziehbar.
- Visualisierung kann direkt auf Reports aufbauen (bereits begonnen) und schrittweise wachsen, ohne den Pipeline-Kern umzubauen.

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
