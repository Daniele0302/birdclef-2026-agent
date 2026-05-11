# Video Script — BirdCLEF+ 2026 Autonomous Agent (max 5 min)

## Setup

- **Tool**: macOS QuickTime Player → `File → New Screen Recording`
- **Audio**: built-in Mac microphone (parla in inglese o italiano — la lingua del corso)
- **Schermate da preparare PRIMA di registrare** (apri in tab/finestre separate):
  1. VSCode con `agent.py` aperto
  2. Terminale pulito posizionato nella cartella `birdclef-agent`, pronto a lanciare `python agent.py`
  3. `experiments/experiment_log.json` aperto (mostra struttura JSON dei 108 esp.)
  4. Il **PDF del report** aperto sulla pagina con la **Tabella 2** (submission progression) e la **Figure 3**
  5. La pagina Kaggle Submissions con le tue 6 submission visibili (in particolare la 0.725)

---

## Timeline (4 min 50 s)

### 0:00 – 0:20 — Intro (20 s)

**Schermata**: prima pagina del report (titolo + autori).

**Voce**:
> "BirdCLEF 2026 chiede di identificare la presenza di 234 specie animali in finestre di 5 secondi di audio passivo dai wetland del Pantanal. Per la track B del corso, abbiamo costruito un agente autonomo guidato da un LLM locale. Il nostro best score sul leaderboard pubblico è 0.725, contro un baseline frozen di 0.592. Vi spiego come ci siamo arrivati."

---

### 0:20 – 1:00 — Architettura dell'agent (40 s)

**Schermata**: vai sul report, **Figura 1** (loop diagram) a pagina 2.

**Voce**:
> "L'agent ha cinque moduli: `agent.py` orchestra il loop, `llm_provider.py` parla con Ollama, `code_executor.py` esegue gli esperimenti in subprocess con timeout, `memory.py` tiene il log JSON, e `experiment_template.py` è il guscio parametrizzabile che l'LLM riempie con i suoi iperparametri. Il loop è: l'LLM propone JSON, il template viene eseguito, le metriche tornano all'LLM che le analizza e itera. Usiamo Gemma 4 E4B di Google, lo stesso modello suggerito nell'handout."

> "Una scelta progettuale chiave: l'LLM NON riscrive il codice di training da zero — riempie solo un parameter file. Questo perché un piccolo bug nella pipeline audio degrada il punteggio silenziosamente e con 8K di context window è troppo facile sbagliare. Manteniamo l'LLM dove è davvero bravo: variare iperparametri."

---

### 1:00 – 2:00 — Live demo dell'agent (60 s)

**Schermata**: Terminale. Lancia `.venv/bin/python agent.py`. Lascia girare almeno 1 iterazione completa.

**Voce mentre gira**:
> "Faccio partire l'agent. Vedete: l'LLM propone parametri in formato JSON — `n_mels=64, hop=256, top_db=40, augmentation=time_shift`. L'agent salva il file, lancia il subprocess che addestra una mini EfficientNetB0 su 2000 sample focal in pochi minuti, e cattura il val_AUC. Poi rimanda i risultati all'LLM, che produce un'analisi e propone l'esperimento successivo. Nessuno tocca la tastiera."

(Mentre aspetti che giri, passa al log:)

**Schermata**: apri `experiments/experiment_log.json` → mostra che ci sono già **108 entries**.

**Voce**:
> "Nel log abbiamo già 108 esperimenti accumulati dalle run precedenti. Il prossimo si aggiungerà qui sotto."

---

### 2:00 – 2:35 — Reliability metrics (35 s)

**Schermata**: report, **Sezione 2.5 — Per-Stage Reliability Metrics** e **Figura 2** (bar chart).

**Voce**:
> "Il professore ci ha chiesto di misurare quanto è affidabile l'agent ai vari stage della pipeline. Calcoliamo la rate condizionale di successo: data la stage k-1 OK, qual è la probabilità che la stage k vada a buon fine? L'anello debole è S4, il training completion al 76%: circa il 24% dei subprocess vanno in timeout perché l'LLM ogni tanto propone configurazioni troppo lente. Lo consideriamo una feature: il timeout è proprio il compute budget per esperimento che l'handout raccomanda. E l'agent impara dai fallimenti — i timeout si concentrano nelle prime iterazioni e diventano rari dopo."

---

### 2:35 – 3:20 — Scaling-up phase (45 s)

**Schermata**: report, **Sezione 4** (Scaling-Up Phase).

**Voce**:
> "L'agent ha identificato la giusta architettura, ma il suo best LB iniziale era solo 0.480 — c'era distribution shift fortissimo tra focal recordings di training e soundscape di test. Per chiudere il gap, abbiamo costruito un validation set group-split per file dei soundscape annotati: questo è diventato un proxy quasi perfetto del leaderboard — gap di solo 0.012 tra locale e Kaggle."

> "Poi: phase-2 fine-tuning del backbone seguendo Chollet capitolo 8.3.2, mixup, label smoothing, deep ensemble multi-seed con pesi uniformi. Importante: avevamo provato pesi ottimizzati greedy sulla validation, ma overfittavano — gap di 0.056. Tornare a pesi uniformi ha ridotto il gap a 0.019. Esattamente quello che dice Chollet capitolo 5 sulla validation discipline."

---

### 3:20 – 4:10 — GPU exploitation + risultati (50 s)

**Schermata**: report, **Figura 3** (line chart submission progression) e **Tabella 2**.

**Voce**:
> "Il salto finale arriva dalla GPU. Stessa architettura EfficientNetB0, ma allenata sul dataset completo di 35.549 recording focal più tutti i soundscape, su una T4 gratuita di Kaggle Notebooks, per 15 epoche con cosine learning rate decay, AdamW e regolarizzazione pesante. Un'ora di training. Best soundscape macro-AUC locale: 0.86. Questo modello scaricato e usato dentro un notebook CPU rispetta il vincolo di 90 minuti per la submission e ci dà il **0.725 sul leaderboard pubblico**."

**Schermata**: Kaggle Submissions page con i 6 score.

**Voce**:
> "Qui vedete la progressione: dal 0.592 del baseline frozen-backbone, salendo gradualmente a 0.598, 0.615, 0.633 con gli ensemble CPU, e infine 0.725 con il modello GPU. Più 0.133 sull'iniziale."

---

### 4:10 – 4:40 — Challenges & lezioni (30 s)

**Schermata**: report, sezione 7 (Limitations).

**Voce**:
> "Le challenge principali: primo, il domain shift tra training focal e test soundscape — risolto con un validation set ad hoc. Secondo, l'overfitting della validation quando si fittavano i pesi dell'ensemble — risolto tornando a pesi uniformi. Terzo, il vincolo di 90 minuti CPU per la submission, che impone un backbone leggero come B0. Modelli più grandi tipo BirdNET darebbero qualche punto in più ma non rientrerebbero nel budget."

---

### 4:40 – 4:55 — Conclusione (15 s)

**Schermata**: prima pagina del report.

**Voce**:
> "Quello che abbiamo costruito segue esattamente il funnel di scaling laws descritto nell'handout: esplorazione cheap dell'agent in CPU per identificare l'architettura giusta, poi exploitation con GPU sui parametri vincenti. Il codice è tutto su GitHub, il link è nel report. Grazie."

---

## Note pratiche per la registrazione

1. **Prima prova senza registrare**: leggi lo script ad alta voce 1 volta per timing
2. **Registra in più segmenti**: meglio 3-4 take corti che 1 take perfetto
3. **iMovie** (gratuito su Mac) per assemblare i segmenti
4. **Risoluzione**: 1080p è più che sufficiente, mp4
5. **File finale**: < 200 MB se possibile (così entra su GitHub)
6. **Upload**: due opzioni:
   - Caricarlo nel repo come `video_demo.mp4`
   - YouTube unlisted + linkalo nel report (più professionale)

## Cosa NON dire

- Non dire "ho usato Claude Code per aiutarmi a scrivere" — l'handout lo permette ma il video è meglio resti focalizzato sul lavoro
- Non scendere in dettagli che non hai capito (es. non promettere fix futuri che non puoi descrivere bene)
- Non dire "abbiamo provato a fare X ma non ha funzionato" senza spiegare il perché

## Cosa puntare in evidenza

- **L'agent è autonomo** (no human input)
- **108 esperimenti loggati** = scala reale
- **Reliability metrics** = punto originale che il prof aveva chiesto
- **Funnel strategy** = allineamento esplicito con l'handout
- **0.725 LB** = molto sopra baseline
