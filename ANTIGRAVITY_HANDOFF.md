# AudioBench — handoff per Antigravity

> **Leggi questo file prima di modificare il progetto.** È il documento di contesto principale: descrive cosa stiamo costruendo, cosa è già implementato, cosa è stato verificato e come proseguire senza confondere le funzionalità reali con quelle future.

## 1. In breve

**AudioBench** è un piccolo banco di test automatici per confrontare render audio offline e rilevare regressioni negli algoritmi DSP o nelle catene di processamento.

Caso d'uso: uno sviluppatore ha un render di riferimento prodotto da una versione nota come corretta e un secondo render prodotto dalla versione che sta modificando. AudioBench confronta i due file, calcola metriche riproducibili, applica soglie configurabili e produce un report HTML leggibile da una persona e un risultato JSON utilizzabile da script o CI.

Il progetto nasce come portfolio tecnico per un tirocinio in audio software engineering, DSP, plugin audio o R&D. Deve privilegiare correttezza, riproducibilità, testabilità e documentazione, non soltanto un'interfaccia appariscente.

## 2. Obiettivo finale

Evolvere da un comparatore di due file WAV a un **regression test bench affidabile e automatizzabile** per algoritmi audio. Un utente dovrebbe poter:

1. generare o preparare segnali di test noti;
2. confrontare output di riferimento e output sotto test;
3. stabilire soglie di qualità appropriate al proprio algoritmo;
4. eseguire più confronti in modo ripetibile;
5. leggere report per ogni caso e un riepilogo della suite;
6. interrompere una pipeline CI se una regressione supera le soglie;
7. in una fase successiva, collegare l'analizzatore a un workflow di render di un plugin o di una catena audio.

**Il primo obiettivo di sviluppo consigliato è una modalità batch pilotata da manifest JSON.** È una progressione concreta dal codice attuale, utile e dimostrabile, senza introdurre subito la complessità di ospitare plugin nativi.

## 3. Perimetro attuale: cosa fa e cosa non fa

### Implementato
- CLI Python con i sottocomandi `generate`, `compare` e `batch`.
- Generazione deterministica di cinque segnali WAV: impulso, sinusoide a 1 kHz, sweep logaritmico, rumore bianco con seed e silenzio.
- Caricamento/scrittura WAV con dati convertiti a float per l'analisi.
- Rifiuto esplicito dei file vuoti, dei campioni non finiti, dei sample rate differenti e dei numeri di canali incompatibili; non viene fatto resampling automatico.
- Metriche di durata, sample peak, RMS, errore RMS, SNR, errore relativo, correlazione, massimo errore assoluto, differenza media tra magnitudini spettrali, errore di forma spettrale normalizzato per livello e stima del ritardo tramite cross-correlazione.
- Quality gate separati per fedeltà, latenza stimata e durata.
- Suite batch pilotate da file manifest JSON con risoluzione percorsi relativi, override soglie per singolo test e isolamento dei guasti (fault isolation).
- Stati complessivi `PASS`, `WARN` o `FAIL`.
- Report HTML autosufficienti per singolo test e dashboard riassuntiva della suite (`summary.html`), con waveform SVG e JSON incorporato.
- Output JSON autonomo per singolo test e aggregato per suite (`summary.json`).
- Codici di uscita CLI: `0` per PASS/WARN (salvo `--fail-on-warn`), `1` per FAIL/ERROR o WARN con quell'opzione, `2` per input/configurazione/manifest/errore runtime gestito.
- Suite estesa di 37 test unitari in `tests/` (`test_audio.py`, `test_metrics.py`, `test_batch.py`, `test_cli.py`).
- GitHub Actions per eseguire i test su Python 3.10, 3.11, 3.12 e 3.13.

### Non implementato (non dichiararlo come se lo fosse)

- Non carica né esegue direttamente plugin VST3, Audio Unit o altri plugin nativi.
- Non controlla una DAW e non orchestra il rendering di plugin.
- Non fa resampling automatico.
- Non implementa una misura certificata/oversampled di true peak.
- Non implementa ancora un analizzatore calibrato THD+N.
- Le metriche spettrali correnti sono indicatori diagnostici basati sul confronto delle magnitudini; non sono una risposta in frequenza calibrata.
- La stima del ritardo via cross-correlazione può essere ambigua su silenzio, sinusoidi pure e segnali periodici. Va preferito un impulso o un segnale broadband appropriato per il test di latenza.
- Un risultato PASS non equivale a una certificazione percettiva, a una prova che il suono sia musicalmente migliore o alla conformità a uno standard.

Questi limiti sono importanti e vanno mantenuti espliciti nella documentazione.

## 4. Struttura del repository

```text
AudioBench/
├── .github/workflows/tests.yml   # CI: installa pacchetto ed esegue pytest
├── AGENTS.md                     # istruzioni persistenti per l'agente di coding
├── ANTIGRAVITY_HANDOFF.md        # questo documento: contesto e piano
├── README.md                     # documentazione progetto in inglese
├── LICENSE                       # licenza MIT
├── pyproject.toml                # packaging, dipendenze e comando CLI
├── examples/
│   ├── README.md
│   └── demo/                     # WAV, report HTML e JSON dimostrativi
├── scripts/
│   └── create_demo.py            # rigenera le demo PASS/FAIL
├── src/audiobench/
│   ├── __init__.py                # versione del pacchetto
│   ├── __main__.py                # supporto a python -m audiobench
│   ├── audio.py                   # I/O WAV e segnali di test
│   ├── cli.py                     # interfaccia a riga di comando
│   ├── metrics.py                 # metriche, stima ritardo e soglie
│   └── report.py                  # report HTML autosufficiente
└── tests/
    ├── test_audio.py
    └── test_metrics.py
```

### Architettura

```text
WAV di riferimento + WAV da verificare
                 │
                 ▼
          audio.py (I/O)
                 │
                 ▼
         metrics.py (misure)
                 │
                 ▼
       valutazione delle soglie
                 │
        ┌────────┴────────┐
        ▼                 ▼
   report.py             JSON
    HTML                  │
        └────────┬────────┘
                 ▼
              cli.py
```

Tenere la logica DSP indipendente dalla CLI e dal report. I calcoli devono essere testabili direttamente con array NumPy senza necessità di una UI o di file reali.

## 5. Cosa è stato fatto finora

La prima implementazione è stata realizzata come MVP Python con una struttura di pacchetto installabile:

- `audio.py`: funzioni di I/O e generazione ripetibile dei segnali di test.
- `metrics.py`: calcolo metriche, stima del ritardo con cross-correlazione e applicazione delle soglie.
- `report.py`: report HTML autonomo, senza CDN o risorse esterne obbligatorie.
- `cli.py`: comandi CLI, validazione iniziale degli argomenti, scrittura dei report e codici di uscita utili in CI.
- `tests/`: test per determinismo dei segnali, round-trip WAV, render identici, errore di gain, segno del ritardo, mismatch dei canali, durata diversa e metriche spettrali.
- `.github/workflows/tests.yml`: configurazione per eseguire i test su più versioni di Python.
- `scripts/create_demo.py` e `examples/demo/`: dimostrazioni riproducibili di un confronto esatto (PASS) e di un render con ritardo artificiale di 2,5 ms (FAIL).

### Verifica eseguita

Durante la preparazione di questo handoff è stato eseguito, dalla root del repository:

```bash
python -m pytest -q
```

Risultato osservato in questo ambiente: **9 test superati** con Python 3.13.5. Questa è una verifica locale; non significa che la GitHub Actions sia stata eseguita su tutte le versioni Python. Esegui nuovamente i test dopo ogni modifica.

## 6. Stack e installazione

- Python >= 3.10
- NumPy
- SciPy
- SoundFile
- pytest come dipendenza di sviluppo
- setuptools per packaging

Installazione consigliata in un ambiente virtuale:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -e ".[dev]"
python -m pytest -q
```

Comandi principali:

```bash
python -m audiobench --help
python -m audiobench generate --help
python -m audiobench compare --help

python -m audiobench generate \
  --output-dir examples/test_signals \
  --sample-rate 48000 \
  --duration 1
```

Esempio di confronto:

```bash
python -m audiobench compare \
  --expected reference.wav \
  --actual render.wav \
  --report reports/comparison.html \
  --json reports/comparison.json
```

Rigenera anche le demo incluse con:

```bash
python scripts/create_demo.py
```

## 7. Come interpretare le metriche

- **SNR rispetto al riferimento**: rapporto tra RMS del riferimento e RMS dell'errore. Se i due file sono identici campione per campione, il programma rappresenta il risultato come match esatto invece di serializzare un infinito numerico.
- **RMS error / error dBFS**: entità assoluta della differenza campione per campione.
- **Estimated delay**: ritardo in campioni e millisecondi stimato tramite cross-correlazione. Un valore positivo indica che il segnale `actual` è in ritardo rispetto a `expected`.
- **Duration delta**: durata effettiva di `actual` meno durata di `expected`.
- **Correlation**: somiglianza della forma dei campioni, non un punteggio percettivo.
- **Mean spectral magnitude delta**: differenza media in dB fra magnitudini spettrali; diagnostica, non risposta in frequenza calibrata.
- **Gain-normalized spectral shape error**: confronto delle forme spettrali dopo aver normalizzato il livello RMS; utile per distinguere parzialmente cambi di livello da cambi di forma.

Le soglie predefinite sono valori iniziali generici (SNR minimo 60 dB, ritardo massimo 0,5 ms, differenza di durata massima 1 ms, errore RMS massimo -80 dBFS quando il riferimento è silenzioso). Non vanno assunte universali: ogni protocollo di test dovrebbe motivarle in base all'algoritmo e al segnale usato.

L'allineamento stimato è riportato come diagnostica, ma i quality gate di fedeltà sono deliberatamente calcolati sui file non riallineati: compensare il delay prima di valutare la fedeltà potrebbe nascondere una vera regressione di latenza.

## 8. Milestone completata: batch da manifest JSON

È stato implementato il sottocomando `batch`:

```bash
python -m audiobench batch \
  --manifest examples/demo/demo_manifest.json \
  --output-dir reports/batch
```

### Funzionalità implementate e verificate

1. Lettura manifest JSON con nome della suite, soglie globali e lista di test case (`src/audiobench/batch.py`).
2. Risoluzione deterministica dei percorsi audio relativi alla directory del manifest (`resolve_path`).
3. Merge ed override granulare delle soglie per singolo test (`merge_thresholds`).
4. Isolamento dei guasti (fault isolation): un file mancante, corrotto o con sample rate non corrispondente viene registrato come `ERROR` senza bloccare l'esecuzione della suite.
5. Reporting completo: artifact individuali `<test_id>.html` e `<test_id>.json`, unione machine-readable in `summary.json` e dashboard visiva in `summary.html` (`src/audiobench/report.py`).
6. Codici di uscita conformi POSIX/CI: `0` (tutti PASS/WARN), `1` (FAIL o ERROR, oppure WARN con `--fail-on-warn`), `2` (manifest non valido, JSON errato o errore di configurazione).
7. Suite di test estesa a 37 unit test (`test_batch.py`, `test_cli.py`, `test_audio.py`, `test_metrics.py`).
8. Retrocompatibilità completa per i sottocomandi `generate` e `compare`.

## 9. Roadmap successiva

Dopo il batch mode, procedere possibilmente in quest'ordine:

1. **Risposta in frequenza per bande**: usare segnali di test noti (sweep o rumore) e una metodologia documentata per stimare le deviazioni.
2. **THD+N**: protocollo basato su sinusoide, finestre e gestione delle componenti spettrali con test sintetici noti. Indicare limiti e non dichiarare conformità a standard senza averla verificata.
3. **Benchmark storico**: confrontare risultati JSON fra commit/versioni e rendere visibili le regressioni.
4. **Adapter di rendering**: solo dopo aver deciso un host e una piattaforma specifici. Possibili strade: harness di test per un plugin controllato dal progetto, oppure integrazione con un host CLI già disponibile. Non tentare subito di supportare ogni plugin/DAW.
5. **UI opzionale**: valutare una UI soltanto se facilita il workflow; la CLI deve rimanere utile da sola.

## 10. Regole per Antigravity

- Esamina il repository prima di cambiare file e usa il codice esistente come fonte di verità.
- Implementa una milestone alla volta e mantieni la portata realistica.
- Non riscrivere tutto il progetto per aggiungere una funzionalità circoscritta.
- Non aggiungere framework pesanti, una UI o dipendenze non necessarie senza motivazione.
- Non inventare metriche o nomi “professionali” per misure che non implementano davvero un protocollo valido.
- Mantieni la differenza fra misure diagnostiche, gate di regressione e misure standardizzate esplicita nella documentazione.
- Per ogni nuova metrica, crea test con segnali sintetici di cui conosci la risposta attesa.
- Per ogni feature CLI, testa input validi, input invalidi, file mancanti, codici di uscita e output generati.
- Non ridurre le soglie o indebolire i test semplicemente per far passare una suite.
- Aggiorna `README.md` e questo handoff quando una funzione cambia stato da proposta a implementata.
- Alla fine di ogni attività, esegui `python -m pytest -q`, riporta esattamente i test passati/falliti e riassumi i file modificati.
- Se un test non può essere eseguito per mancanza di rete/dipendenze o limitazioni dell'ambiente, dichiararlo invece di affermare che è passato.

## 11. Primo prompt da dare all'agente

Puoi usare questo prompt dopo aver aperto la cartella `AudioBench` in Antigravity:

> Leggi prima `AGENTS.md`, `ANTIGRAVITY_HANDOFF.md` e `README.md`, poi esplora il codice attuale e lancia i test esistenti. Non cambiare codice finché non hai riassunto architettura, limiti e stato dei test. Dopo questa verifica, implementa come prossima milestone il batch mode basato su manifest JSON descritta nella sezione 8 dell'handoff. Mantieni retrocompatibile `compare`, riusa `metrics.py` e `evaluate_thresholds`, aggiungi test completi per manifest, percorsi, override soglie, report e exit code, aggiorna la documentazione e rigenera le demo. Non implementare hosting VST/AU o UI in questa milestone. Alla fine esegui tutta la suite e riporta risultati e file modificati.
