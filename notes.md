# Related work zu Historischer Semantik / Word Embeddings

## Projekte

| Projekt      |Beschreibung                                                                | Link |
| -------------|---------------------------------------------------------------------------|:-------:| 
| JeSeMe    |                                                   | http://jeseme.org/|
| HistWords |                                                   | https://github.com/williamleif/histwords|
| HistorEx  |                                                   | https://www.fiz-karlsruhe.de/sites/default/files/FIZ/Dokumente/Forschung/ISE/Publications/2019-ESWC-D-HistorEx-Exploring-Historical-Text-Corpora.pdf|
| Wortschatz Leipzig|                                                   |https://wortschatz.uni-leipzig.de/de |
| WittFind  |                                                   | http://wittfind.cis.uni-muenchen.de/|
 



## Daten

| Projekt      |Beschreibung                                                                | Link |
| -------------|---------------------------------------------------------------------------|:-------:| 
|Australian Newspaper Online|                                                   | http://anno.onb.ac.at/ |
| verdi-requiem   |     Historische Zeitungsartikel über die Messa da Requiem von Giuseppe Verdi                                               | https://github.com/torstenroeder/verdi-requiem| 
|Wiki Source     |                                                   | https://de.wikisource.org/wiki/Zeitschriften_(Musik) |
|     |                                                   | |
|     |                                                   | |

## Dornseiff
| **pro** |**contra**|
|:---------|:----------|
|viel zitiertes onomasiologisches Lexikon|  |
|Korpusbasiert, nicht mehr subjektiv (im Gegensatz zu vorherigen Auflagen) |Wörter, die seltener als 10 mal im Leipziger Wortschatz vorkommen, wurden aus Dornseiff entfernt  |
|Basiert auf Zeitschriften des Leipziger Wortschatzes aus dem Zeitraum 1987-2001 | |
|Eigene Hauptgruppe (10) für "Fühlen, Affekte, Charaktereigenschaften"  |Kritik: kein reines Emotionslexikon, häufig sind relevante Wörter im Lexikon verstreut (außerhalb HG 10) |
|Wortgruppen: Liebe, Hass, Zorn, Hoffnung, Wählerisch, Wunsch, Reizbar, Charakter |Trauer fehlt |
|     |    |


Aufbau eines Wörterbuchartikels:
- Titel
- Verweisteil (enger semantischer Zusammenhang zu anderen Artikeln)
- Formenteil 
    - Affixe (optional)
    - Ausrufe (optional)
    - Substantive
    - Adjektive / Adverben
    - Verben

## HistWords  
- Preprocessing: Lowercasing, Punktuation entfernen, Stoppwörter entfernen
- Hyperparameter: Windowsize 4, Embeddingsize 300
- Alignment der W2V-Embeddings mit orthogonal Procrustes
- Zwei Wege, um Semantic Change zu erkennen: 
    1. Paarweise Wortähnlichkeiten zwischen Wörtern über Zeit berechnen (Test von Hypothesen über spezifische Wörter)
    2. Zeigen, wie ein individuelles Embedding sich über Zeit verschiebt/verändert (Cos-sim des gleichen Wortes über Zeiträume hinweg)
- Erkenntnis: Häufige Wörter ändern ihre Bedeutung langsamer
- Visualisierung: 
    1. Berechne die Menge aller ähnlichen Wörter eines Wortes über Zeiten hinweg (k-nearest neighbors)
    2. Berechne die t-SNE Word Embeddings dieser Wörter für die aktuellste Zeitperiode
    3. Berechne für jede vorherige Zeitperiode das t-SNE Embedding für das Zielwort   
    --> die Background-Wörter um das Zielwort herum sind in ihrer aktuellen Position gezeigt und verändern sich nicht

## Beobachtungen
- Wort "künstlich" früher im Sinne "kunstvoll" (anno-Korpus), heute (leipzig-Korpus) näher an "nicht natürlich"
- Wort "ausdrucksvoll" im dta Korpus im Sinne von Mimik, im anno-Korpus stärker auf Musik bezogen
- Wort "stimmungsvoll": anno: stark auf Musik bezogen, dta: auf Wetter bezogen, leipzig: auf Feiern und Beisammensein bezogen


## Ausblick

- Ähnlichkeiten nach Wortarten gruppieren
- Stemming, Lemmatisierung
- Mehrwortlexeme?
- Macht es Sinn, Doc2Vec einzusetzen? Oder Passagen der Zeitschriften zu betrachten?
- NER, CRF (nltk)
- Lassen sich die unteschiedlichen Schreibweisen bestimmten Korpora/Zeitschriften zuordnen?
- Macht es Sinn, Schreibweisen zusammenzufassen? 
	- contrapunktisch/kontrapunktisch
	- effectvoll/effektvoll
