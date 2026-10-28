# Istruzioni per gli agenti di coding

Prima di modificare AudioBench, leggere `ANTIGRAVITY_HANDOFF.md` e `README.md`.

## Obiettivo

Sviluppare in modo incrementale un regression test bench Python per render audio offline e algoritmi DSP. Non confondere le funzionalità già implementate con la roadmap.

## Regole obbligatorie

- Ispezionare prima l'implementazione esistente; evitare riscritture ampie senza necessità.
- Mantenere separati I/O, metriche DSP, quality gate, report e CLI.
- Conservare la compatibilità del comando `compare` salvo che una modifica incompatibile sia espressamente richiesta e documentata.
- Ogni nuova misura DSP deve avere test sintetici con un comportamento atteso noto.
- Ogni nuova funzione CLI deve avere test di input valido/invalid, file mancante, output e exit code.
- Non fare resampling silenzioso e non abbassare soglie o indebolire test per ottenere PASS.
- Non descrivere la stima diagnostica come misura calibrata, certificata o conforme a standard se non è stato implementato e verificato il protocollo corrispondente.
- Preferire dipendenze leggere; motivare l'aggiunta di ogni nuova dipendenza.
- Aggiornare la documentazione insieme al codice.
- Non dichiarare test eseguiti se non lo sono stati.

## Comandi di verifica

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m audiobench --help
python scripts/create_demo.py
```

## Milestone attesa

Vedi la sezione “Prossimo milestone suggerito: batch da manifest JSON” in `ANTIGRAVITY_HANDOFF.md`. La prossima milestone prevista è il batch mode per coppie di render già esistenti; **non** è ancora l'automazione del rendering di plugin VST3/AU.
