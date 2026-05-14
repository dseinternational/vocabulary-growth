---
title: "Crescita del vocabolario — panoramica sullo stato del progetto"
subtitle: "Fino a VG09B (ancorato)"
date: "2026-05-14"
lang: it
format:
  pdf:
    documentclass: article
    papersize: a4
    fontsize: 11pt
    mainfont: "Source Sans 3"
    sansfont: "Source Sans 3"
    monofont: "Monaspace Neon"
    monofontoptions: "Scale=0.8125"
    linestretch: 1.25
    geometry:
      - top=25mm
      - left=25mm
      - right=25mm
      - bottom=25mm
    colorlinks: true
    link-citations: true
    toc: true
    toc-depth: 2
    number-sections: true
    number-depth: 3
  html:
    toc: true
    toc-depth: 2
    number-sections: true
    number-depth: 3
---

::: {.callout-warning}
Questo documento è stato redatto con l'assistenza di un modello di intelligenza
artificiale (Claude, Anthropic) e dovrebbe essere verificato in modo indipendente
prima di essere utilizzato come base per conclusioni di ricerca. La presente
versione è la traduzione italiana di un originale in inglese; in caso di
discrepanze fa fede l'originale.
:::

## Sintesi direzionale

### A cosa serve il progetto

Famiglie, insegnanti, logopedisti e clinici che seguono bambini con
sindrome di Down (SD) si trovano regolarmente a dover rispondere a
domande pratiche sullo sviluppo del vocabolario: _«A che età un bambino
tipico con SD comprende 100 parole? A che età ne dice 100? Il mio bambino
di 3 anni capisce molto ma dice molto poco — rientra nei valori
attesi?»_ Le decisioni cliniche ed educative che ne derivano — quando
inviare a uno specialista, cosa aspettarsi, come fissare gli obiettivi,
se il progresso di un bambino è inusuale — dipendono dal disporre di
risposte affidabili a queste domande.

Questo progetto si propone di fornire un insieme di statistiche
_interrogabili_ in grado di informare aspettative, intervento e pratica
didattica per bambini con SD tra circa 12 mesi e 7 anni di età:

- quante parole un bambino tipico con SD _comprende_ a ogni età,
- quante un bambino tipico con SD _dice_ a ogni età, e
- come queste due quantità crescono insieme.

Le quantità corrispondenti sono stimate anche per bambini a sviluppo
tipico (ST), prevalentemente come punto di riferimento — il confronto
qualitativo _SD-vs-ST_ è la parte che conta di più per la
comunicazione clinica.

### Mettere in comune l'evidenza internazionale

Nessuno studio singolo sullo sviluppo del vocabolario nella SD è
abbastanza ampio da sostenere stime stabili e risolte per età sull'intero
intervallo di 12–90 mesi. Il contributo distintivo del progetto consiste
nel **riunire i dati di più studi internazionali in un'unica analisi
statistica coerente**: 964 righe con età valida da 510 soggetti SD
unici, distribuite su **10 etichette di studio** che coprono Regno Unito,
Irlanda, Italia e Stati Uniti, armonizzate in un unico dataset pooled.
Per gli attuali modelli congiunti compreso + parlato, 950 di queste righe
portano almeno uno dei due esiti modellati.

È proprio questa messa in comune che rende possibile l'analisi. È anche
ciò che crea il problema metodologico centrale che il progetto ha dovuto
risolvere: gli studi differiscono sistematicamente in _quali bambini
reclutano_, _quale modulo CDI somministrano_ e _a quali età raccolgono i
dati_. Un pooling ingenuo produce traiettorie di popolazione che
seguono quali studi hanno contribuito a quali età, anziché come i
bambini si sviluppino effettivamente. L'intera iterazione di modellazione
documentata di seguito è, in sostanza, la storia di come abbiamo
identificato e assorbito quei confondenti tra studi e tra soggetti
mantenendo intatto il segnale evolutivo.

In parallelo viene mantenuto un riferimento ST: un campione Wordbank
riproducibile al 10 % (1.655 osservazioni) viene fittato con la stessa
famiglia di modelli, in modo che il contrasto SD-vs-ST sia
metodologicamente omogeneo.

### Perché le risposte sono restituite come distribuzioni di probabilità

L'inferenza statistica bayesiana restituisce un'intera distribuzione di
probabilità a posteriori su ciascuna quantità di interesse, anziché un
singolo numero con un intervallo di confidenza. È questa cornice
probabilistica a sostenere affermazioni del tipo _«c'è una probabilità
dell'80 % che un bambino tipico di 3 anni con sindrome di Down comprenda
fra X e Y parole»_ — un fraseggio che molti lettori già danno per
scontato che un intervallo di confidenza supporti, mentre solo gli
intervalli bayesiani effettivamente lo fanno.

La stessa distribuzione a posteriori sostiene anche distribuzioni
_predittive_ per un singolo bambino non osservato, distribuzioni di
_milestone_ («a che età il bambino tipico con SD raggiungerà le 100
parole prodotte?») e distribuzioni a _comprensione equiparata_ («a 200
parole comprese, quale frazione dice il bambino tipico con SD?»). Tutti e
tre i framing compaiono in questo rapporto.

### Quadro scientifico principale (stabile attraverso le iterazioni del modello)

- La crescita del vocabolario nella SD prosegue lungo l'intero
  intervallo di età campionato (≈ 12–90 mesi); la traiettoria tipica
  della SD è monotona e crescente.
- La produzione orale resta indietro rispetto alla comprensione in modo
  persistente — il divario è più ampio e più duraturo nella SD che nello
  sviluppo ST e _cresce_ con il livello di comprensione nelle fasce di
  età iniziali e intermedie, prima di restringersi nella fascia di età
  superiore (anche se ciò potrebbe essere attribuibile al soffitto dello
  strumento).
- L'eterogeneità tra soggetti sul rapporto di produzione
  `q = p_S / p_U` è la singola fonte di variazione più grande nella
  famiglia di modelli ($\tau^{\text{subj}}_q \approx 1{,}19$ sulla scala
  logit sotto VG09B), sostanzialmente più grande della variazione tra
  studi.

### Dove si trova attualmente la modellazione

La modellazione ha iterato su dieci specifiche — da una baseline
univariata «età → parlato» (VG01) fino a un modello SD congiunto
compreso + parlato con effetti casuali di studio e per-soggetto su
entrambe le traiettorie (VG09) — e una decima, **VG09B**, che risolve
un problema strutturale di identificabilità presente in VG09 ancorando
la correzione del processo gaussiano a zero in corrispondenza di un'età
di riferimento.

**VG09B è il candidato attuale a sostituire VG09 come modello SD
congiunto di riferimento:** le sue diagnostiche sono pulite
($\hat R \le 1{,}01$, $\text{ESS} \ge 400$ su tutti i parametri
riportati), preserva la partizione di varianza di VG09 e la
parametrizzazione rimuove la cresta GP–intercetta che produceva gli
avvertimenti marginali su $\hat R$ ed ESS in VG09. Il costo è che la
traiettoria del rapporto di produzione alle età intermedie si sposta
verso il basso di circa 5–10 punti percentuali rispetto a VG09 — uno
spostamento nella direzione sostenuta dall'argomento strutturale, non un
artefatto di reporting.

Restano aperte alcune questioni metodologiche: il calo non monotono di
`q` oltre i ~72 mesi sotto VG09 e VG09B (assente in VG07), quanto
l'interpretazione delle età superiori sia influenzata dai soffitti
finiti delle checklist e se l'ancoraggio del GP debba essere applicato
simmetricamente al resto della famiglia di modelli (VG05–VG08 e gli
univariati VG01–VG04). Un filone di lavoro separato — portare il
vocabolario segnato/gestuale nella famiglia di modelli — è anch'esso
delineato di seguito. Tutto ciò si riflette in [Prossimi
passi](#prossimi-passi).

## Dati e metodi

### Dataset

La fonte dati SD contiene attualmente 964 righe con età valida da 510
ID soggetto unici, distribuite su 10 etichette di studio internazionali
nella vista DuckDB `vocab_combined`. Gli studi contribuenti coprono Regno
Unito, Irlanda, Italia e Stati Uniti e sono stati raccolti nell'arco di
circa tre decenni. Utilizzano moduli CDI / MacArthur diversi e
protocolli di studio diversi.

Per i modelli SD bivariati, 950 righe presentano almeno una delle
osservazioni `understood` o `spoken` (704 righe `understood`, 949 righe
`spoken`, 703 coppie complete compreso + parlato). 288 dei 510 soggetti
SD hanno ≥ 2 righe di analisi bivariata, contribuendo per 728 delle 950
righe (massimo 8 su un singolo soggetto). Questa struttura a misure
ripetute entro soggetto motiva il passaggio a intercette casuali di
soggetto in VG08 e VG09.

Il riferimento ST utilizza un campione riproducibile al 10 % di Wordbank
(1.655 osservazioni), fittato con la stessa famiglia di modelli affinché
i contrasti SD-vs-ST siano metodologicamente coerenti.

### Approccio di modellazione

Tutti i modelli condividono la stessa forma statistica:

- una traiettoria media liscia (ma flessibile) sull'età, composta da una
  tendenza lineare più una correzione di processo gaussiano nello spazio
  di Hilbert (HSGP) sulla scala logit;
- una verosimiglianza Beta-Binomiale con dispersione variabile con l'età
  — Beta-Binomiale anziché Binomiale perché bambini individuali alla
  stessa età mostrano una variabilità molto maggiore di quella che una
  Binomiale può assorbire;
- (nei modelli congiunti) una componente di rapporto di produzione
  accoppiata, in modo che `p_S ≤ p_U` sia garantito per costruzione;
- (da VG07 in poi) intercette casuali sulla scala logit per assorbire
  scarti sistematici a livello di studio e di soggetto che altrimenti
  contaminerebbero la traiettoria di popolazione.

Gli esiti sono conteggi riferiti dai genitori, ricavati da checklist
finite di tipo CDI / MacArthur. Le probabilità del modello si
riferiscono pertanto alla probabilità che una _parola della checklist_
sia compresa o detta, e i conteggi attesi sono parole attese all'interno
dell'inventario somministrato, non conoscenza lessicale totale. Quando i
bambini si avvicinano all'estremo superiore di un modulo, parole
ulteriori al di fuori della checklist restano non osservate, quindi
apparenti appiattimenti, incertezza compressa o rapporti di produzione
molto elevati possono in parte riflettere il soffitto dello strumento
anziché una saturazione evolutiva.

L'inferenza è realizzata tramite Monte Carlo hamiltoniano (PyMC +
nutpie / NUTS). La configurazione di campionamento per la _reportistica_
è 6 catene × (6.000 tune + 6.000 draw) con `target_accept = 0,95`. La
soglia di convergenza del progetto è $\hat R \le 1{,}01$ e
$\text{ESS} \ge 400$ per ogni parametro riportato; nella corsa di
reportistica del 12 maggio 2026, 252.000 draw di VG01–VG07 hanno
prodotto una divergenza (in VG06) e soddisfatto la soglia in ogni
modello.

## Famiglia di modelli

| ID    | Esito                          | Popolazione | Effetti casuali                                             |
| ----- | ------------------------------ | ----------- | ----------------------------------------------------------- |
| VG01  | Parlato                        | SD          | —                                                           |
| VG02  | Compreso                       | SD          | —                                                           |
| VG03  | Parlato                        | ST          | —                                                           |
| VG04  | Compreso                       | ST          | —                                                           |
| VG05  | Compreso + parlato (congiunto) | SD          | —                                                           |
| VG06  | Compreso + parlato (congiunto) | ST          | —                                                           |
| VG07  | Compreso + parlato (congiunto) | SD          | Studio                                                      |
| VG08  | Compreso + parlato (congiunto) | SD          | Studio + soggetto (su U)                                    |
| VG09  | Compreso + parlato (congiunto) | SD          | Studio + soggetto (su U e q)                                |
| VG09B | Compreso + parlato (congiunto) | SD          | Come VG09, con GP ancorato e prior più stretti sugli anchor |

VG09B è la variante sperimentale che implementa «Opzione A + D» dalla
nota sulle opzioni strutturali
(`notes/202605131500-vg09-structural-options.md`):

- **Opzione A (prior più stretti sugli anchor di q):**
  `p_slope_low_q ~ Beta(3, 22)` (media ≈ 0,12, sd ≈ 0,06);
  `p_slope_hi_q ~ Beta(20, 4)` (media ≈ 0,83, sd ≈ 0,08).
- **Opzione D (ancoraggio per-draw del GP):** `g_u` e `g_q` sono
  riparametrizzati come `eta · (g_unit − g_unit(a_ref))` con
  `a_ref = 54` mesi. Ogni draw a posteriori del GP passa per zero
  all'età di riferimento, così la tendenza lineare definisce
  univocamente il livello in quel punto e il GP può descrivere solo
  deviazioni dalla linearità.

Tutte le altre componenti (effetti casuali di studio, effetti casuali di
soggetto su `u` e `q`, prior di dispersione, verosimiglianza
Beta-Binomiale) sono identiche a VG09.

## Iterazione attraverso la famiglia — cosa ci ha insegnato ogni passo

### VG01–VG06 — traiettorie di baseline

I modelli SD univariati (VG01 parlato, VG02 compreso) e le loro
controparti ST (VG03, VG04) hanno stabilito che i dati sono
abbastanza informativi a ogni età da identificare una traiettoria di
crescita liscia, e che la dispersione Beta-Binomiale si comporta bene
sull'intero intervallo di età. I modelli congiunti (VG05 SD, VG06 ST)
hanno aggiunto un rapporto di produzione accoppiato, che impone
`p_S ≤ p_U` e rende il divario comprensione–produzione un parametro di
prima classe anziché una quantità derivata.

Il modello SD congiunto VG05 mostrava già il quadro qualitativo
SD-vs-ST che è sopravvissuto a tutti i raffinamenti successivi:

- comprensione e produzione SD crescono entrambe sull'intero intervallo
  di età, ma la produzione resta indietro rispetto alla comprensione di
  anni anziché mesi;
- il rapporto di produzione `q` sale da prossimo a zero a prossimo a uno
  su un intervallo di comprensione molto più ampio rispetto alla ST — il
  divario è _strutturalmente più ampio_ nella SD, non semplicemente
  ritardato.

VG05 ha anche messo in evidenza un problema: un apparente calo di parole
_comprese_ fra circa 40 e 60 mesi che il GP fittava nonostante fosse
evolutivamente improbabile.

### La scoperta del paradosso di Simpson — VG07 corregge l'artefatto

Un'indagine (`notes/202604121055-understood-ds-decline.md`) ha
ricondotto il calo del vocabolario compreso a 40–60 mesi a uno
_spostamento di composizione_ negli studi contribuenti: gli studi con
punteggi più alti (1, 2, 6, 7) smettono di contribuire dati sulle parole
comprese dopo ~50 mesi, mentre quelli con punteggi più bassi (3, 5)
proseguono. La traiettoria della miscela pooled cala anche se nessuna
traiettoria di studio singolo cala. È un caso da manuale di paradosso di
Simpson.

**VG07** ha introdotto intercette casuali a livello di studio sia sul
logit del compreso sia sul logit del rapporto di produzione, con prior
$\tau_U, \tau_q \sim \text{HalfNormal}(0{,}5)$. Le SD a posteriori tra
studi sono sostanziali ($\tau_U \approx 0{,}50$,
$\tau_q \approx 0{,}66$ sulla scala logit) e ben identificate
(ESS > 7.000). Con quegli scarti di studio assorbiti, la traiettoria del
compreso a livello di popolazione diventa monotona crescente sull'intero
intervallo di età. L'«uncino» a sinistra nei grafici
compreso-vs-parlato scompare.

Questo ci porta a concludere che _quello che sembrava un declino evolutivo
era guidato da quali studi avevano contribuito dati a quali età_. È
ciò che motiva la preferenza per VG07 rispetto a VG05 per qualsiasi
reportistica SD che tocchi l'intervallo del compreso 40–70 mesi.

### VG08 — intercette casuali di soggetto sul compreso

VG07 trattava ancora ogni osservazione come se provenisse da un bambino
indipendente. Ma 288 dei 510 soggetti SD hanno ≥ 2 osservazioni di
analisi bivariata, contribuendo per 728 delle 950 righe usate dai
modelli SD congiunti. Il reclutamento longitudinale tende a
sovra-rappresentare le famiglie più ingaggiate, perciò quei soggetti
con osservazioni multiple si collocano al di sopra della media di
popolazione — e contribuiscono in modo sproporzionato alla
verosimiglianza.

VG08 ha aggiunto un'intercetta casuale di soggetto non centrata sul
logit del compreso. La partizione di varianza si è separata in modo
pulito:

- SD tra studi sul compreso ($\tau_U$): essenzialmente invariata
  (~0,51) — a conferma che gli effetti casuali di studio in VG07 _non_
  stavano segretamente assorbendo la correlazione entro soggetto;
- SD tra soggetti sul compreso
  ($\tau^{\text{subj}}_U \approx 0{,}78$): la singola componente di
  variazione più grande emersa finora;
- la dispersione Beta-Binomiale residua all'età intermedia è scesa da
  `exp(1,75) ≈ 5,8` a `exp(2,93) ≈ 18,6` — cioè a circa un terzo di
  quanto stimava VG07.

La traiettoria si è spostata in modi clinicamente significativi: più
bassa a 12–18 mesi, più ripida attraverso la fascia di età intermedia, e
con un _plateau_ del rapporto di produzione attorno a `q ≈ 0,78–0,84`
da ~60 mesi in poi, anziché la salita continua verso 1,0 di VG07.

### VG09 — intercette casuali di soggetto anche sul rapporto di produzione

Se la variabilità tra soggetti giustifica un effetto casuale di soggetto
sul compreso, la stessa logica si applica a quanto di ciascun vocabolario
compreso del bambino venga effettivamente prodotto. VG09 ha aggiunto un
effetto casuale di soggetto parallelo sul logit del rapporto di
produzione.

Questo ha messo in luce la componente di varianza più grande dell'intera
famiglia di modelli: $\tau^{\text{subj}}_q \approx 1{,}20$ sulla scala
logit, equivalente sulla scala dei conteggi a un `q` del bambino SD
tipico che si colloca all'incirca su un intervallo cinque volte più
ampio attorno alla mediana di popolazione a comprensione equiparata.
VG08 stava assorbendo tutto ciò nella dispersione Beta-Binomiale lato
parlato; con VG09, la sovradispersione residua sul vocabolario parlato è
crollata in modo sostanziale — $\exp(a_{\kappa_S})$ è salito da ~6,4
(VG08) a ~27,6 (VG09), una riduzione di un ordine di grandezza.

Tre confronti leave-one-subject-out (LOSO) hanno stabilito che VG09
generalizza meglio dei suoi predecessori. Il gold standard — un
confronto K = 5 LOSO con re-fit — ordina VG09 > VG08 > VG07 con
significatività statistica schiacciante:

| Coppia       | elpd_diff (migliore − peggiore) |  dSE | diff / dSE |
| ------------ | ------------------------------: | ---: | ---------: |
| VG09 vs VG07 |                      **+339,0** | 34,6 |   **9,79** |
| VG09 vs VG08 |                          +109,4 | 19,3 |       5,68 |
| VG08 vs VG07 |                          +229,6 | 26,1 |       8,80 |

(Fonte: `output/comparisons/kfold_loso_compare.csv`. n = 510 soggetti;
15 re-fit a configurazione `test`, 41 minuti totali di wall time.)

VG09 vince sia descrittivamente (partizione di varianza più pulita) sia
predittivamente (miglior K-fold LOSO).

### Problema diagnostico di VG09 — e la risposta strutturale

Al campionamento di qualità da reportistica, VG09 ha restituito cinque
parametri con $\hat R$ marginale sopra 1,01 (massimo 1,020) e uno con
$\text{ess}_{\text{tail}} < 400$ (minimo 358). Stringere il sampler
(`target_accept` = 0,99, `tune` = 8.000) ha raddoppiato il wall time e
_non ha migliorato_ la diagnostica: si tratta di un problema di
geometria a posteriori, non di passo di campionamento.

La diagnosi strutturale (`notes/202605131400-vg09-sampler-diagnostics.md`
e `notes/202605131500-vg09-structural-options.md`): la traiettoria di
`q` in VG09 ha tre componenti che portano ciascuna un livello globale —
la tendenza lineare (`intercept_q + slope_q · a_z`), la correzione HSGP
(`eta_q · g_unit_q`) e le famiglie di intercette casuali. I dati
identificano la loro _somma_, non la loro decomposizione, perciò la
posteriori ha forma di cresta lungo quella direzione. L'HSGP ha media
zero _in attesa a priori_ (mediata sui draw), ma ogni singolo draw della
funzione GP porta una costante arbitraria che compete con
`intercept_q`.

VG09B implementa la più piccola correzione strutturale credibile: prior
più stretti sugli anchor e un ancoraggio per-draw del GP a zero
all'età `a_ref = 54` mesi. Il resto di questo rapporto si concentra su
ciò che VG09B modifica e ciò che non modifica.

## VG09B — risultati dettagliati

### Diagnostiche: ogni parametro segnalato è rientrato

La variante A+D di VG09B — prior più stretti sugli anchor di q più un
ancoraggio del GP a `a_ref = 54` mesi — ha eliminato i flag diagnostici
riportati di VG09. L'ESS bulk è all'incirca raddoppiato sui parametri
precedentemente segnalati e $\hat R$ è sceso a $\le 1{,}009$ su ogni
parametro riportato (soglia: $\hat R \le 1{,}01$,
$\text{ESS} \ge 400$). Ciò è coerente con la diagnosi strutturale per cui
VG09 aveva una ridondanza GP–intercetta, ma il miglioramento diagnostico
va attribuito al cambiamento combinato A+D, non all'ancoraggio del GP
isolatamente.

| Parametro       | VG09 $\hat R$ | VG09 $\text{ESS}_{\text{bulk}}$ | VG09B $\hat R$ | VG09B $\text{ESS}_{\text{bulk}}$ |
| --------------- | ------------- | ------------------------------- | -------------- | -------------------------------- |
| `slope_q`       | 1,020         | 430                             | **1,008**      | **1.244**                        |
| `p_slope_low_q` | 1,013         | 1.008                           | **1,007**      | **1.610**                        |
| `p_slope_hi_q`  | 1,012         | 431                             | **1,006**      | **1.221**                        |
| `eta_q`         | 1,010         | 483                             | **1,009**      | **1.207**                        |
| `p_slope_hi_u`  | 1,014         | 628                             | **1,007**      | **1.338**                        |
| `intercept_u`   | 1,012         | 759                             | **1,007**      | **1.225**                        |
| `intercept_q`   | 1,005         | 799                             | **1,004**      | **2.152**                        |

Sintesi: _$\hat R \le 1{,}01$ ed $\text{ESS} \ge 400$ sui parametri
riportati._

### La partizione di varianza è preservata

Lo scopo di VG09B era di correggere la parametrizzazione di VG09
_senza_ modificare la partizione di varianza sostanziale. Le componenti
strutturali di varianza sono essenzialmente identiche a quelle di VG09:

| Componente (SD su logit)                        | VG07 | VG08 | VG09 | **VG09B** |
| ----------------------------------------------- | ---: | ---: | ---: | --------: |
| Tra studi, compreso ($\tau_U$)                  | 0,50 | 0,51 | 0,52 |  **0,52** |
| Tra studi, q ($\tau_q$)                         | 0,66 | 0,74 | 0,94 |  **0,99** |
| Tra soggetti, compreso ($\tau^{\text{subj}}_U$) |    — | 0,78 | 0,84 |  **0,84** |
| Tra soggetti, q ($\tau^{\text{subj}}_q$)        |    — |    — | 1,20 |  **1,19** |
| Dispersione BB compreso $a_{\kappa_U}$ (log)    | 1,75 | 2,93 | 3,10 |  **3,10** |
| Dispersione BB parlato $a_{\kappa_S}$ (log)     | 1,40 | 1,86 | 3,32 |  **3,31** |

La variazione tra soggetti su `q` rimane la singola componente più
grande della famiglia, di un ordine di grandezza maggiore del nugget
Beta-Binomiale su entrambi gli esiti. VG09B eredita intatto questo
risultato.

### Sintesi a posteriori (parametri selezionati)

| Parametro              | Media |   SD | HDI 90 %       | $\text{ESS}_{\text{bulk}}$ | $\hat R$ |
| ---------------------- | ----: | ---: | -------------- | -------------------------: | -------: |
| $\text{intercept}_U$   | −1,15 | 0,22 | [−1,52, −0,80] |                      1.225 |    1,007 |
| $\text{slope}_U$       |  1,13 | 0,20 | [ 0,80, 1,45]  |                        930 |    1,007 |
| $\eta_U$               |  0,58 | 0,16 | [ 0,32, 0,83]  |                      3.524 |    1,002 |
| $\text{intercept}_q$   | −0,23 | 0,31 | [−0,73, 0,27]  |                      2.152 |    1,004 |
| $\text{slope}_q$       |  1,53 | 0,31 | [ 1,03, 1,99]  |                      1.244 |    1,008 |
| $\eta_q$               |  0,94 | 0,24 | [ 0,56, 1,35]  |                      1.207 |    1,009 |
| $\tau_U$               |  0,52 | 0,14 | [ 0,31, 0,73]  |                     10.507 |    1,000 |
| $\tau_q$               |  0,99 | 0,21 | [ 0,67, 1,32]  |                      6.900 |    1,000 |
| $\tau^{\text{subj}}_U$ |  0,84 | 0,04 | [ 0,77, 0,91]  |                      5.831 |    1,000 |
| $\tau^{\text{subj}}_q$ |  1,19 | 0,09 | [ 1,05, 1,33]  |                      6.354 |    1,000 |
| $a_{\kappa_U}$         |  3,10 | 0,11 | [ 2,93, 3,28]  |                     13.068 |    1,001 |
| $a_{\kappa_S}$         |  3,31 | 0,14 | [ 3,09, 3,53]  |                      8.730 |    1,001 |

(Fonte:
`output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/diagnostics.csv`.)

La riparametrizzazione ha spostato `intercept_q` da −1,17 (VG09) a
−0,23 (VG09B) — uno spostamento verso l'alto di circa 0,94 unità logit
— e `p_slope_hi_q` da 0,80 a 0,93. $\tau^{\text{subj}}_q$ ed `eta_q`
sono essenzialmente invariati. Lo spostamento è quindi interamente tra
il GP e la tendenza lineare, _non_ tra il GP e gli effetti casuali:
esattamente ciò che l'argomento strutturale prediceva.

### Traiettorie: bambino SD tipico (mediano) sotto VG09B

Traiettoria a livello di popolazione — effetti casuali di studio e di
soggetto posti a zero, cioè il cambiamento atteso entro soggetto man
mano che un bambino SD tipico invecchia. I conteggi di vocabolario sono
qui aspettative del modello su un totale di 800 parole (`Ey_median`); le
HDI al 90 % (`Ey_hdi_lo`–`Ey_hdi_hi`) riflettono l'incertezza sulla
traiettoria SD tipica, non la dispersione tra singoli bambini. Alle età
maggiori, in particolare oltre i ~72 mesi, queste vanno lette come stime
entro la checklist. Un bambino può continuare ad apprendere parole al di
fuori dell'inventario somministrato anche quando il conteggio
compreso o parlato basato sulla checklist si avvicina al soffitto del
modulo.

| Età (mesi) | Compreso | HDI 90 %   | Parlato | HDI 90 %       | q (mediana) | HDI 90 %       |
| ---------: | -------: | ---------- | ------: | -------------- | ----------- | -------------- |
|         12 |       20 | [13, 27]   |     0,1 | [0,04, 0,26]   | 0,007       | [0,002, 0,013] |
|         18 |       43 | [32, 57]   |     0,9 | [0,40, 1,56]   | 0,021       | [0,010, 0,034] |
|         24 |       87 | [64, 111]  |     4,2 | [1,99, 6,67]   | 0,048       | [0,025, 0,074] |
|         30 |      140 | [105, 174] |    12,3 | [6,25, 19,00]  | 0,089       | [0,050, 0,132] |
|         36 |      183 | [140, 225] |    33,2 | [18,0, 48,6]   | 0,182       | [0,112, 0,260] |
|         42 |      218 | [170, 266] |    78,4 | [49,5, 109,2]  | 0,363       | [0,248, 0,481] |
|         48 |      255 | [201, 308] |   134,8 | [93,7, 175,8]  | 0,534       | [0,408, 0,654] |
|         54 |      298 | [241, 357] |   192,1 | [145,3, 244,0] | 0,651       | [0,535, 0,759] |
|         60 |      344 | [281, 408] |   258,0 | [201,2, 314,1] | 0,757       | [0,648, 0,851] |
|         66 |      392 | [324, 462] |   320,9 | [258,7, 383,0] | 0,829       | [0,722, 0,922] |
|         72 |      435 | [361, 510] |   362,1 | [294,2, 428,3] | 0,844       | [0,721, 0,949] |
|         78 |      473 | [389, 555] |   380,3 | [303,8, 456,1] | 0,820       | [0,661, 0,962] |
|         84 |      506 | [417, 599] |   398,1 | [307,3, 482,0] | 0,804       | [0,621, 0,984] |
|         90 |      536 | [440, 632] |   432,3 | [339,1, 521,7] | 0,825       | [0,645, 0,996] |

(Fonte:
`output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/posterior_summary_u.csv`,
`posterior_summary_s.csv`, `posterior_summary_q.csv`.)

![Traiettoria congiunta VG09B: vocabolario compreso e parlato mediani per
età per un bambino tipico con SD, con bande HDI al 90 % per la
traiettoria SD tipica.
](../output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/joint_trajectory_hdi.png){#fig-vg09b-joint
fig-align="center" width=85%}

![Rapporto di produzione VG09B `q = p_S / p_U` rispetto all'età (SD).
](../output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/production_rate.png){#fig-vg09b-q
fig-align="center" width=85%}

### Tasso di produzione rispetto alle parole comprese

Per famiglie e clinici un confronto importante è quello a _comprensione
equiparata_: a parità di vocabolario compreso da un bambino, quale
frazione tipicamente produce? La curva di VG09B è più ripida di quella
di VG07 al centro dell'intervallo di comprensione — cioè la produzione
recupera su una finestra di comprensione più stretta una volta che
inizia a recuperare — ma parte più tardi. Questo accentua la divergenza
SD–ST, già sostanziale, oltre le ~150 parole comprese.

Anche qui, `q = p_S / p_U` significa «la frazione di parole della
checklist comprese che sono anche dette». In prossimità del soffitto
della checklist, il vocabolario recettivo non può più crescere
all'interno dello strumento anche se un bambino continua ad apprendere
parole al di fuori, perciò le stime di `q` nella fascia superiore vanno
interpretate come rapporti di produzione entro inventario, non come
rapporti sull'intero vocabolario del bambino.

![VG09B: tasso di produzione rispetto alle parole comprese (SD).
](../output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/production_rate_by_understood.png){#fig-vg09b-q-vs-u
fig-align="center" width=85%}

### Il ritardo di produzione: SD (VG09B) vs ST (VG06)

La versione cronologica dello stesso confronto pone la traiettoria del
rapporto di produzione SD sotto VG09B accanto al riferimento ST sotto
VG06. Il contrasto è netto: un bambino ST tipico attraversa `q = 0,5`
(dicendo metà delle parole che comprende) attorno ai 17 mesi e raggiunge
`q = 0,9` entro i 23 mesi. Un bambino SD tipico sotto VG09B attraversa
`q = 0,5` a ~47 mesi — un ritardo cronologico di circa due anni e mezzo
— e non raggiunge affatto `q = 0,9` all'interno dell'intervallo di età
campionato. La curva SD raggiunge il picco a ~0,84 a 72 mesi e ridiscende
a ~0,82–0,83.

![Rapporto di produzione `q` rispetto all'età — SD (VG09B) vs ST
(VG06). Le bande sono HDI al 90 % rispettivamente per la traiettoria SD
tipica e ST tipica. Le linee orizzontali tratteggiate segnano
`q = 0,5` e `q = 0,9`.
](../output/comparisons/ds_td_q_by_age_vg09b.png){#fig-ds-td-q-age
fig-align="center" width=85%}

Attraversamenti di milestone approssimati (traiettoria mediana):

| Milestone  | ST (VG06) | SD (VG09B)    | Ritardo SD                      |
| ---------- | --------- | ------------- | ------------------------------- |
| `q = 0,25` | 13,6 mesi | 38,2 mesi     | ~25 mesi                        |
| `q = 0,50` | 16,7 mesi | 46,8 mesi     | ~30 mesi                        |
| `q = 0,75` | 19,7 mesi | 59,6 mesi     | ~40 mesi                        |
| `q = 0,90` | 22,7 mesi | non raggiunto | oltre l'intervallo del campione |

Una visione complementare a comprensione equiparata che utilizzi VG09B
(il modello SD congiunto di riferimento) invece di VG09 mantiene la
storia qualitativa SD-vs-ST ma sposta la curva SD verso il basso a
comprensione bassa–intermedia. Gli attraversamenti SD si spostano verso
valori di comprensione più alti rispetto ai numeri basati su VG09 del
documento di review della riunione.

![Rapporto di produzione `q` rispetto alle parole comprese — SD (VG09B)
vs ST (VG06). La curva SD si colloca decisamente a destra di quella ST,
indicando che i bambini SD accumulano sostanzialmente più vocabolario
recettivo prima che il loro vocabolario parlato recuperi.
](../output/comparisons/ds_td_q_vs_understood_vg09b.png){#fig-ds-td-q-u
fig-align="center" width=85%}

Entrambe le viste mostrano lo stesso risultato in framing diversi:
l'asimmetria comprensione–produzione nella SD non è un semplice ritardo
cronologico — è _strutturalmente più ampia e più duratura_ rispetto
all'asimmetria ST, su entrambi gli assi di età e di comprensione.

### Confronto a tre vie di q: VG07, VG09, VG09B

Il singolo grafico più importante di questo rapporto mostra come la
traiettoria del rapporto di produzione cambi attraverso i tre modelli SD
in competizione. VG07 è la baseline rilevante perché non ha effetti
casuali di soggetto su `q` e non è pertanto soggetto alla ridondanza
GP/intercetta/effetti casuali che ha motivato VG09B.

![Rapporto di produzione `q` rispetto all'età, SD: VG07 (solo effetti
casuali di studio) vs VG09 (effetti casuali di studio + soggetto su U e
q, GP non ancorato) vs VG09B (come VG09, GP ancorato e prior più
stretti).
](../output/comparisons/vg07_vg09_vg09b_q_by_age.png){#fig-q-three-way
fig-align="center" width=85%}

| Età (mesi) | q VG07 | q VG09 | **q VG09B** |
| ---------: | -----: | -----: | ----------: |
|         12 |  0,041 |  0,011 |   **0,007** |
|         18 |  0,085 |  0,033 |   **0,021** |
|         24 |  0,148 |  0,073 |   **0,048** |
|         30 |  0,211 |  0,133 |   **0,089** |
|         36 |  0,300 |  0,260 |   **0,182** |
|         42 |  0,443 |  0,474 |   **0,363** |
|         48 |  0,599 |  0,657 |   **0,534** |
|         54 |  0,711 |  0,769 |   **0,651** |
|         60 |  0,782 |  0,846 |   **0,757** |
|         66 |  0,833 |  0,890 |   **0,829** |
|         72 |  0,871 |  0,898 |   **0,844** |
|         78 |  0,904 |  0,882 |   **0,820** |
|         84 |  0,934 |  0,870 |   **0,804** |
|         90 |  0,958 |  0,882 |   **0,825** |

(Fonte: `output/comparisons/vg07_vg09_vg09b_q_by_age.csv`.)

Quattro pattern meritano di essere evidenziati:

1. **VG07 è monotono crescente sull'intero intervallo di età** (0,04 a
   12 mesi a 0,96 a 90 mesi), e quella monotonia è la forma attesa
   dalla letteratura CDI pubblicata sulla SD.
2. **VG09 e VG09B raggiungono entrambi un picco poi calano.** VG09
   raggiunge il picco a ~0,90 a 72 mesi e cala a ~0,88 a 90 mesi;
   VG09B raggiunge il picco a ~0,84 a 72 mesi e cala a ~0,82 a 90
   mesi. La coda non monotona compare dopo l'aggiunta degli effetti
   casuali di soggetto su `q`: dove i dati sono sparsi la media di
   popolazione può essere attratta verso il prior, e l'interazione
   inverse-logit / Jensen con l'ampia distribuzione degli effetti
   casuali di soggetto può piegare la curva di popolazione verso il
   basso. Il calo è condiviso da VG09 e VG09B — cioè non è causato
   dalla cresta GP–intercetta che VG09B era stato progettato per
   correggere. VG07 è il comparatore monotono.
3. **Alle età intermedie (36–66 mesi) VG09B si colloca circa 10 punti
   percentuali sotto VG07 e 10–13 punti percentuali sotto VG09.** A 48
   mesi, VG07 / VG09 / VG09B danno 0,60 / 0,66 / 0,53. Lo spostamento
   VG09 → VG09B è coerente con la correzione strutturale A+D: in VG09
   il `g_q` non ancorato portava ~1,5 unità logit di livello costante
   all'età di riferimento che i dati non potevano distinguere da
   `intercept_q + slope_q · a_z(a_ref)`. Una volta forzato il GP a
   passare per zero a 54 mesi, la tendenza lineare definisce
   univocamente il livello lì e il GP è ristretto a deviazioni genuine
   dalla linearità; anche i prior più stretti sugli anchor di q
   contribuiscono al cambiamento di traiettoria.
4. **Alle età molto giovani (12–30 mesi) VG07 è sostanzialmente più
   alto sia di VG09 sia di VG09B.** Questo è un effetto collaterale
   degli effetti casuali di soggetto su `q`: con un'ampia
   distribuzione a livello di soggetto nello spazio logit-`q`, il `q`
   latente a livello di popolazione all'estremo inferiore
   dell'intervallo dei dati si contrae verso zero tramite lo stesso
   effetto Jensen che produce il calo nella coda superiore.

Ci sono _due spostamenti distinti_ nascosti nel confronto a tre vie.
**VG07 → VG09** è l'effetto dell'aggiunta degli effetti casuali di
soggetto su `q`; quello spostamento si manifesta soprattutto
all'estremo giovane (contrazione verso zero a 12–30 mesi) e nella coda
superiore (il plateau / calo dopo i 72 mesi). **VG09 → VG09B** è la
correzione della parametrizzazione; quello spostamento si manifesta
soprattutto alle età intermedie (lo spostamento verso il basso di ~5–10
pp), perché è lì che la ridondanza a livello GP veniva assorbita. VG09B
_non_ è «VG09 fatto sembrare VG07» — è «VG09 con l'identificabilità
latente a livello GP risolta», e la differenza rimanente fra VG09B e
VG07 è il vero effetto dell'aggiunta degli effetti casuali di soggetto
su `q`.

## Estendere la famiglia di modelli: vocabolario segnato

Un passo naturale successivo è portare il vocabolario _segnato_ nella
famiglia di modelli. I bambini con sindrome di Down utilizzano
frequentemente segni o gesti come ponte verso la produzione orale, e
genitori ed educatori chiedono regolarmente se il segnare _aggiunga_ al
repertorio espressivo del bambino o _sostituisca_ le parole che
altrimenti produrrebbero. Un modello congiunto che stimi la probabilità
che una parola sia segnata (oltre a compresa e detta) a ogni età ci
permetterebbe di rispondere direttamente a quella domanda. Il problema è
che i dati sul segnare di cui disponiamo non sono semanticamente
uniformi attraverso gli studi.

### Cosa i dati supportano — e l'eterogeneità semantica

La vista DuckDB `vocab_combined` usata dal codice del modello contiene
414 righe con età valida e un conteggio `signed` non nullo, da 236
soggetti unici distribuiti su cinque etichette di studio:

| Etichetta studio | Righe segnate | Soggetti unici | Intervallo di età (mesi) | Righe con signed > 0 | Righe con compreso + parlato |
| ---------------- | ------------: | -------------: | ------------------------ | -------------------: | ---------------------------: |
| `uk_01`          |           218 |            133 | 15–115                   |                   63 |                           29 |
| `uk_02`          |            95 |             58 | 19–56                    |                   95 |                           89 |
| `uk_04`          |            44 |             18 | 18–45                    |                   35 |                           44 |
| `uk_05`          |            46 |             16 | 17–36                    |                   37 |                           46 |
| `uk_06`          |            11 |             11 | 60–115                   |                   11 |                           11 |

Il conteggio precedente di 101 righe a volte citato dal CSV piatto
`data/vocab_data_merged.csv` non è la base corretta per la modellazione:
quella fusione CSV elimina le colonne di segnare di UK 01 e UK 02 prima
della concatenazione, lasciando solo gli studi 6, 7 e 9. La vista DuckDB
è la fonte autoritativa per i modelli fittati.

Delle 414 righe segnate nella vista DuckDB, 219 righe da 122 soggetti
portano anche conteggi sia `understood` sia `spoken`, perciò un'analisi
congiunta compreso + parlato + segnato è meccanicamente fattibile su
quel sottoinsieme.

Il quadro aggregato (ignorando per un momento la questione semantica)
per quelle 219 righe complete è:

| Banda di età (mesi) | Righe | Righe con signed > 0 | Mediana segnato | Mediana parlato | Mediana compreso |
| ------------------- | ----: | -------------------: | --------------: | --------------: | ---------------: |
| 0–24                |    42 |                   21 |             0,5 |               2 |               60 |
| 24–36               |    90 |                   80 |            48,5 |              16 |            180,5 |
| 36–48               |    49 |                   36 |             102 |              79 |              299 |
| 48–60               |    26 |                   26 |             209 |             284 |              465 |
| 60–72               |     5 |                    5 |             387 |             399 |              654 |
| 72–84               |     3 |                    3 |              23 |             400 |              576 |
| 84–120              |     4 |                    4 |              66 |             391 |              616 |

Due pattern a livello di riga:

- Nella finestra 24–36 mesi — dove si concentra la maggior parte dei
  dati sul segnare — i conteggi segnati superano quelli parlati sulla
  stessa riga di circa 3:1 (mediana 48,5 segnati vs 16 parlati).
- Sull'intero sottoinsieme con segnare completo, 123 delle 219 righe
  hanno `signed > spoken`; la quota media a livello di riga
  `signed / (signed + spoken)` è ~0,55.

Ma questi numeri aggregati nascondono una **mancata corrispondenza
semantica** nel modo in cui il «segnato» è codificato da ciascuno studio
contribuente. Il notebook condiviso di preparazione dati
(`dsegroup/research-data-analysis/projects/vocabulary/notebooks/n000-data-preparation.ipynb`)
rende ciò esplicito:

| Studio fonte                 | Cosa registra `signed`                                                                                                                                                                                                                           | Costruzione                                                                                                                                                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UK 01 (EDG, 1990–2000)       | **Solo-segnato** — parole segnate ma _non_ parlate. Il modulo originale codifica ogni parola con esattamente uno fra `c` (comprende), `v` (dice), `s` (segna), perciò le categorie sono mutuamente esclusive a livello di parola.                | `signed = sum(noun*s, …, verb*s)`, dove le colonne `*s` contano parole con solo il codice «segna». `produced = spoken + signed` è quindi il totale espressivo. La vista DuckDB porta questi conteggi segnati; il CSV piatto no. |
| UK 02 (follow-on EDG)        | **Segnato totale** — parole segnate, con o senza essere anche parlate. La fonte ha colonne esplicite `signed_only`, `signed_spoken`, `spoken_only`, ma la vista DuckDB attuale mantiene solo l'aggregato `signed = signed_only + signed_spoken`. | Totali per riga da `signed_only` + `signed_spoken`; la decomposizione è presente nel CSV sorgente ma persa in `vocab_combined`.                                                                                                 |
| UK 04 (Mason-Apps; studio 6) | Colonna sorgente `signs`. Definizione non documentata nel codice di prep; provvisoriamente trattata come **segnato totale**. Fornisce 44 righe.                                                                                                  | `signed = round(signs)` direttamente dalla fonte.                                                                                                                                                                               |
| UK 05 (Seager; studio 7)     | Colonna sorgente `signed`. Definizione non documentata nel codice di prep; provvisoriamente trattata come **segnato totale**. Fornisce 46 righe.                                                                                                 | `signed = signed` copia diretta.                                                                                                                                                                                                |
| UK 06 (RLI; studio 9)        | **Segnato totale** — la colonna SPSS è `CheckUnderAndSign`, «(parole) comprese e segnate». Conta parole segnate indipendentemente dal fatto che siano anche parlate. Fornisce 11 righe.                                                          | `signed = CheckUnderAndSign` dopo una rinomina.                                                                                                                                                                                 |

L'implicazione principale: un valore `signed = 50` da UK 01 significa
«50 parole segnate ma non parlate»; un valore `signed = 50` da UK 06
significa «50 parole segnate, possibilmente anche parlate». I due **non
sono sulla stessa scala**. La definizione «solo-segnato» è un
sottoinsieme stretto della definizione «segnato totale»; la differenza è
la sovrapposizione «segnato e parlato».

Una seconda limitazione aggrava la prima: la famiglia di modelli
attuale usa una sola dimensione di inventario `N` per riga per definire
il tasso di successo Beta-Binomiale. La vista DuckDB porta attualmente
`survey_vocab_max` per le righe del segnare (396/690 per UK 01, 800 per
UK 02 e UK 06, 418 per UK 04/05), ma quei denominatori devono essere
verificati e resi espliciti per un modello del segnare. In particolare,
il denominatore di UK 04/05 è ancora marcato come `TODO` in
`scripts/prepare_data.py`, e il CSV piatto non porta `form_max_spoken` /
`form_max_understood` per il sottoinsieme di 101 righe sul segnare.

### Cosa implica la mancata corrispondenza semantica per la modellazione

1. **`signed + spoken` non è interpretabile attraverso gli studi.** Per
   le righe EDG di UK 01 la somma equivale al vocabolario espressivo
   totale (segnato e parlato sono non sovrapposti); per le righe RLI di
   UK 06 la somma conta due volte ogni parola che è sia segnata sia
   parlata. Qualsiasi modello che usi `signed + spoken` come quantità
   derivata produrrà stime di popolazione non interpretabili _a meno
   che_ la convenzione non sia tenuta costante sulle righe in fitting.
2. **`q_G = p_G / p_U` (rapporto di probabilità del segnato) significa
   cose diverse in studi diversi.** Sotto la convenzione solo-segnato,
   `q_G` risponde a _«delle parole che questo bambino comprende, quale
   frazione segna ma non dice?»_ Sotto la convenzione segnato totale,
   `q_G` risponde a _«delle parole che questo bambino comprende, quale
   frazione segna in qualche forma?»_ Sono quantità evolutive diverse —
   la prima raggiunge un picco e cade man mano che il segnare lascia
   spazio al parlato, la seconda non fa che salire (e converge a `q_U`)
   man mano che il bambino acquisisce ogni modalità di ogni parola.
   Mettere insieme righe attraverso le due convenzioni e modellare una
   singola `q_G(a)` produrrebbe una curva senza chiara
   interpretazione.
3. **Il modello congiunto più pulito richiede conteggi decomposti a
   livello per riga.** Quello che vorremmo idealmente avere è, per
   riga, `n_understood_only`, `n_spoken_only`, `n_signed_only`,
   `n_spoken_and_signed`, più la dimensione dell'inventario e la
   convenzione di osservazione. UK 02 ha tutto ciò nella fonte
   originale ma la vista DuckDB lascia cadere la decomposizione. UK 01
   ha una decomposizione mutuamente esclusiva specifica dello
   strumento (le colonne per-parola `*c/*v/*s`), senza una categoria
   osservata «parlato e segnato» sotto quella codifica. UK 04, UK 05,
   UK 06 possono o non possono fornire la decomposizione completa —
   sarebbe necessario verificare i dati di origine sottostanti.

### Implicazioni per le strutture di modello candidate

L'eterogeneità della preparazione dati acuisce i compromessi fra le tre
opzioni delineate in precedenza.

**Opzione S1 — Baseline univariato «età → segnato».** Riflette VG01
(parlato) e VG02 (compreso) con una traiettoria GP + Beta-Binomiale
fittata per le parole segnate. La mancata corrispondenza semantica conta
meno qui che per i modelli congiunti, perché S1 è descrittivo: chiede
solo «dato il conteggio segnato di ciascuna riga, qual è la traiettoria
liscia di parole segnate attese per età?» È ancora difendibile se (a)
lo fittiamo separatamente alle righe «solo-segnato» e «segnato totale»,
producendo due traiettorie distinte; oppure (b) ci limitiamo a una
singola convenzione. Economico da fittare; rivela se la curva liscia in
una qualsiasi delle due convenzioni mostra la forma attesa di
ascesa-poi-caduta (solo-segnato) o di salita monotona (segnato totale).
Utile anche autonomamente per la reportistica clinica.

**Opzione S2 — Modello congiunto trivariato «età → compreso + parlato +
segnato».** Estende la struttura bivariata di VG05/VG07/VG09B a tre
esiti accoppiati:

- `p_U(a)` — probabilità che una parola sia compresa;
- `p_S(a)` — probabilità che una parola sia detta, con `p_S ≤ p_U`
  imposto da `p_S = p_U · q_S(a)`;
- `p_G(a)` — probabilità che una parola sia segnata, di nuovo limitata
  superiormente da `p_U` tramite `p_G = p_U · q_G(a)`.

Il problema semantico è **decisivo** per S2. Se fittiamo S2 su righe
dove `signed` è segnato totale, `q_G` è il rapporto
_segnato-comunque-sia_. Se fittiamo su righe dove `signed` è
solo-segnato, `q_G` è il rapporto _segnato-ma-non-parlato_. Le due
stime non possono essere combinate in un singolo modello coerente
senza (a) restringersi a una sola convenzione, (b) modellare la
convenzione come una variabile categorica per riga con funzioni `q_G`
separate per convenzione, oppure (c) ri-derivare i dati fusi in modo
che la decomposizione (`n_signed_only`, `n_signed_and_spoken`) sia
preservata per riga, modellando poi separatamente ogni componente.

**Opzione S3 — Modello di produzione condizionato alla modalità.»**
Riformula la domanda esplicitamente come _scelta di modalità espressiva
data la produzione di una parola_: per ciascuna parola compresa che il
bambino può produrre, qual è la probabilità che la produca (a) solo
parlata, (b) solo segnata, (c) entrambe, (d) nessuna? Questo è un esito
categorico per parola — un modello multinomiale /
Dirichlet-multinomiale — e ci permetterebbe di tracciare la
_transizione_ da espressione segno-dominante a parlato-dominante come
quantità di prima classe, anziché leggerla a partire da due rapporti
paralleli.

S3 è **il modello che i dati realmente richiedono**, _se_
ri-deriviamo i dati fusi in modo da preservare la decomposizione di
modalità di ciascuno studio. UK 01 EDG fornisce una codifica mutuamente
esclusiva (`*c/*v/*s` a livello per parola); UK 02 registra la
decomposizione a quattro vie in modo esplicito (`signed_only`,
`signed_spoken`, `spoken_only`, `understood_only`). L'attuale vista
`vocab_combined` butta via parte di queste informazioni per UK 02 e non
flagga la convenzione di UK 01. Recuperarla permetterebbe a S3 di
usare le righe in cui la modalità è registrata con semantica onesta e
un modello di osservazione esplicito. Perseguire S3 è quindi
principalmente un investimento di preparazione dati piuttosto che di
plumbing della verosimiglianza.

### Domande pratiche da risolvere prima di fittare

Quattro domande, in ordine:

1. **Ri-derivare il dataset fuso in modo che la semantica del segnare
   sia preservata.** UK 01 e UK 02 codificano informazioni di modalità
   utili al livello dei dati sorgente, ma l'attuale CSV piatto elimina
   le loro colonne di segnare e la vista DuckDB elimina la
   decomposizione di UK 02. Il primo pezzo concreto di lavoro è
   estendere lo schema fuso per portare `n_understood_only`,
   `n_spoken_only`, `n_signed_only`, `n_signed_and_spoken` dove
   disponibili, più un flag per quale definizione segue `signed` per
   ciascuno studio. Senza questo passo, un modello del segnare in
   pooling non poggia su basi interpretabili.
2. **Verificare i dati sorgente di UK 04, UK 05.** Per gli studi 6 e
   7 il codice di prep non documenta se la colonna sorgente `signs` /
   `signed` sia solo-segnato o segnato totale. La risposta dovrebbe
   essere negli strumenti di indagine originali / dizionari dei dati di
   quegli studi. È una domanda da un'ora per qualcuno con accesso ai
   file grezzi; cambia materialmente quali righe possiamo mettere in
   pooling.
3. **Denominatori di inventario.** Verificare ed esporre la dimensione
   dell'inventario CDI / MacArthur per ogni riga di segnare. La vista
   DuckDB ha già `survey_vocab_max` per le attuali righe del segnare,
   ma il valore di 418 per UK 04/05 è ancora marcato come da
   confermare in `scripts/prepare_data.py`, e un modello del segnare
   non dovrebbe dipendere da denominatori impliciti o ambigui.
4. **Confrontabilità tra bande di età.** Anche dopo la correzione
   semantica, la copertura per età è confusa con studio e convenzione
   del segnare: UK 01 copre 15–115 mesi sotto una convenzione
   solo-segnato, UK 02 copre 19–56 mesi con una convenzione segnato
   totale decomponibile, UK 04/05 contribuiscono righe più giovani, e
   UK 06 contribuisce solo 11 righe più anziane con segnato totale.
   Vale la pena esplorare se il prior sulla lengthscale del GP o
   l'intervallo di età della reportistica necessitino di trattamento
   speciale per l'esito segnato.

Il primo esperimento raccomandato, una volta risolte queste questioni,
è un **modello Beta-Binomiale univariato «età → solo-segnato» su righe
in cui i conteggi solo-segnato sono recuperabili** — UK 01
direttamente e UK 02 dopo aver reintrodotto la sua colonna sorgente
`signed_only`. Se quello fitta in modo pulito, S3 su UK 01 + UK 02 è il
naturale secondo passo. Includere UK 04/05/06 richiede o di sapere
quale convenzione seguono o di accettare il costo di modellarle come
convenzioni di osservazione separate con le proprie funzioni `q_G`.

## Conclusioni

1. **Il quadro scientifico sostanziale è stabile attraverso VG07 → VG08
   → VG09 → VG09B.** La crescita del vocabolario SD prosegue
   sull'intero intervallo di età campionato; la produzione orale resta
   indietro rispetto alla comprensione in modo persistente; il divario
   SD–ST è strutturalmente più ampio di un semplice ritardo
   cronologico; la variazione tra soggetti sul rapporto di produzione è
   la singola fonte di eterogeneità più grande della famiglia.
2. **VG09B è un candidato difendibile a sostituire VG09 come modello SD
   di riferimento.** Le diagnostiche sono pulite, la parametrizzazione
   A+D è una risposta di principio alla cresta GP–intercetta, e la
   partizione di varianza è essenzialmente identica a quella di VG09.
   Lo spostamento del rapporto di produzione alle età intermedie è
   compatibile con l'argomento strutturale, ma poiché VG09B ha
   modificato sia i prior degli anchor di q sia l'ancoraggio del GP, va
   riportato come risultato della variante A+D combinata anziché
   attribuito al solo ancoraggio del GP.
3. **La coda non monotona di `q` oltre i ~72 mesi è la principale
   questione interpretativa che rimane.** È condivisa da VG09 e VG09B,
   mentre VG07 rimane monotono, quindi non può essere attribuita alla
   correzione di parametrizzazione A+D. Meccanismi plausibili
   includono l'ampia distribuzione degli effetti casuali di soggetto
   che interagisce con dati sparsi all'estremo superiore
   dell'intervallo di età (Jensen sul logit inverso), e la
   compressione di misurazione mentre i bambini si avvicinano al
   soffitto finito della checklist. Questo richiede un'indagine
   deliberata anziché un aggiustamento dei parametri.
4. **Il rapporto tecnico necessita di aggiornamento per VG09B.** Il
   commit `97023ac` ha già riscritto `docs/report/` nello schema a
   capitoli `vgNN` e ha aggiunto capitoli per VG07, VG08 e VG09. La
   lacuna rimanente del rapporto è VG09B e il testo di
   confronto-modelli / discussione a valle, se VG09B viene promosso.

## Prossimi passi

Ordinati approssimativamente per priorità.

1. **Decidere se promuovere VG09B a VG09 (il nome canonico).** Prove
   per la promozione: diagnostiche pulite, partizione di varianza
   preservata, parametrizzazione strutturalmente onesta. Costo: il
   rapporto tecnico necessita revisione e i valori di `q` riportati
   alle età intermedie si sposteranno.
2. **Se promosso, applicare l'ancoraggio del GP simmetricamente al
   resto della famiglia.** Specificamente: `common.py` per i modelli
   univariati VG01–VG04, `common_bivariate.py` per VG05–VG06 e le
   relative definizioni VG07/VG08 in `common_bivariate_re.py`. Quei
   modelli hanno in linea di principio la stessa cresta GP–intercetta;
   semplicemente non l'abbiamo ancora visto mordere le diagnostiche
   perché hanno meno componenti globali sovrapposte su ciascun esito.
   La coerenza conta più della vittoria marginale sul sampler.
3. **Rieseguire il K-fold LOSO con VG09B.** L'argomento strutturale
   prevede che si collocherà essenzialmente dove si era collocato VG09,
   ma il controllo empirico è in sospeso. 15 re-fit a configurazione
   `test`, ≈ 40 minuti di wall time.
4. **Investigare la coda di q non monotona.** Questo è indipendente
   dalla correzione A+D ed è visibile in VG09 e VG09B ma non in VG07.
   Opzioni da considerare:
   - stringere $\tau^{\text{subj}}_q$ (ad es. `HalfNormal(0,25)`
     invece di 0,5);
   - restringere il prior sulla lengthscale del GP in modo che il GP
     non possa piegarsi su scale decennali;
   - quantificare quanto le osservazioni e le traiettorie a posteriori
     siano vicine ai soffitti rilevanti delle checklist CDI /
     MacArthur;
   - restringere l'intervallo di interrogazione alle età con dati
     sostanziali e segnalare esplicitamente la coda superiore come
     estrapolazione e come stima entro-checklist anziché
     dell'intero-vocabolario.
5. **Aggiornare il rapporto tecnico** (`docs/report/`):
   - aggiungere un capitolo VG09B se il modello viene promosso, o
     documentarlo come sensibilità / modello candidato denominato in
     caso contrario;
   - aggiornare `model-comparison.qmd`, `discussion.qmd` e qualsiasi
     figura/tabella copiata in modo che non mescolino i numeri di
     riferimento di VG09 e VG09B;
   - ri-renderizzare il rapporto dopo la decisione sul modello di
     riferimento.
6. **Affrontare la causa principale dei dati di comprensione mancanti
   nella raccolta dati futura.** 80 osservazioni nella finestra 40–60
   mesi hanno dati di parlato senza dati corrispondenti di compreso,
   prevalentemente dagli studi 1 e 5. La raccolta o armonizzazione dati
   futura dovrebbe dare priorità a coppie complete compreso + parlato
   nell'intervallo 40–70 mesi.
7. **Aprire issue GitHub per le voci sopra.** Il progetto al momento ha
   zero issue aperti; la pianificazione vive in note e descrizioni di
   PR. Per decisioni di questo peso (sostituire VG09 con VG09B,
   propagare l'ancoraggio al resto della famiglia, aggiornamento del
   rapporto VG09B) vogliamo ticket tracciabili.
8. **Definire l'ambito di un'estensione al segnare — partendo dalla
   preparazione dati.** I dati sul segnare in `vocab_combined` sono
   _semanticamente eterogenei_: in alcuni studi contribuenti `signed` è
   segnato-ma-non-parlato, in altri è segnato-indipendentemente. Il
   primo pezzo concreto di lavoro è ri-derivare lo schema fuso in modo
   che la decomposizione di modalità disponibile (`understood_only`,
   `spoken_only`, `signed_only`, `signed_and_spoken`) sia preservata
   per riga e la convenzione del segnare sia segnalata. UK 01 e UK 02
   portano le informazioni chiave a livello sorgente, ma il CSV piatto
   elimina le loro colonne del segnare e la vista DuckDB elimina la
   decomposizione di UK 02. In parallelo, verificare i dati sorgente di
   UK 04 / UK 05 / UK 06 per determinare quale convenzione segue
   ciascuno, e verificare i denominatori d'inventario a livello di
   modulo attualmente esposti come `survey_vocab_max`. _Solo dopo che
   questi passi di preparazione dati saranno stati sistemati_ vale la
   pena codificare le opzioni di modellazione delineate in [Estendere
   la famiglia di modelli](#estendere-la-famiglia-di-modelli-vocabolario-segnato).
   Il primo modello raccomandato è quindi un Beta-Binomiale univariato
   «età → solo-segnato» su righe in cui i conteggi solo-segnato sono
   recuperabili — UK 01 direttamente e UK 02 dopo aver reintrodotto la
   sua colonna sorgente `signed_only`.

## Riferimenti al materiale sorgente

- Nota sui risultati di VG09B:
  `notes/202605141200-vg09b-findings.md`.
- Nota sulle opzioni strutturali che ha motivato VG09B:
  `notes/202605131500-vg09-structural-options.md`.
- Indagine diagnostica di VG09:
  `notes/202605131400-vg09-sampler-diagnostics.md`.
- Documento di review della riunione con la storia più ampia VG07–VG09:
  `notes/202605120945-meeting-project-review.md`.
- Output per-modello di VG09B (trace, figure, tabelle riassuntive):
  `output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/`.
- Dati + figura del confronto a tre vie su q:
  `output/comparisons/vg07_vg09_vg09b_q_by_age.{csv,png,svg}`.
- Confronto K-fold LOSO (VG07 / VG08 / VG09):
  `output/comparisons/kfold_loso_compare.csv`.
- Notebook sorgente di preparazione dati (definisce le convenzioni del
  segnare per studio menzionate nella sezione sull'estensione al
  segnare):
  `dsegroup/research-data-analysis/projects/vocabulary/notebooks/n000-data-preparation.ipynb`.
