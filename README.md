# 19_WE
Training Word Embeddings   

directory structure:
```bash
.
├── analogy.py
├── anno_corpus (not published yet)
│   ├── anno_scraper.py
│   ├── corpus
│   │   ├── input_txt
│   │   └── prep_txt
│   ├── Corpus_description.md
│   └── models
├── dta_corpus
│   ├── corpus
│   │   ├── dta_xml
│   │   ├── input_txt
│   │   └── prep_txt
│   ├── dta2txt.sh
│   ├── get_dta_corpus.txt
│   ├── models
│   └── tei2txt.xsl
├── preprocessing.py
├── README.md
├── notes.md
├── requirements.txt
├── similarity.py
├── training.py
└── visualization.py


```

## 1. Requirements

```shell
pip3 install -r requirements.txt
```

## 2. Data

Data needs to be in raw txt format.  
Save your data in any (new) directory of the project.  
Your data can be stored in one or more files (in the same dir).  
See below for example data (DTA, ANNO-Music Corpus) or use your own data.  

## 3. Preprocess corpus
Preprocess your trainingdata  
Will be saved in one preprocessed sentence per line (nltk sent-tokenize)  

```shell
python3 preprocessing.py <PATH TO INPUT DIR> <PATH TO OUTPUT DIR> -l -p -n -s -o -longs -u -b
```

data will be saved at './<PATH TO OUTPUT DIR\>/NUMBER_OF_PREPROCESSED_CORPUS/data/  
parser arguments will be saved at './\<PATH TO OUTPUT DIR>/NUMBER_OF_PREPROCESSED_CORPUS/metadata.csv'   
 
  
parser options:  

| argument      |description                          | default |
| ------------- |-------------------------------------|:-------:| 
| raw           | input dir with raw txt files        |         | 
| target        | output dir for storing prep txt     |         |
| -l            | lowercasing                         | False   |
| -p            | remove punct                        | False   |
| -n            | remove numbers                      | False   |
| -s            | remove stopwords                    | False   | 
| -o            | remove ocr-errors (like oͤ,uͤ,aͤ)      | False   |
| -longs        | change ſ to s                       | False   |
| -u            | change umlauts to digraphs          | False   |
| -b            | detect bigrams                      | False   |


## 4. Train model
```shell
python3 training.py <MODEL> <PATH/TO/CORPUS> <PATH/TO/MODEL>
```
    
  
parser options:  

| argument      |description                                                                | default |
| ------------- |---------------------------------------------------------------------------|:-------:| 
| model         | 'word2vec' or 'fasttext'                                                  | word2vec| 
| corpus        | path to dir with preprocessed corpus files (one sentence per line)        |         |
| target	    | path to dir for storing model						                        |	      |
| -sg           | network architecture: Skipgram (1), Cbow (0)                              | 0       |
| -hs           | training algorithm: hierarchical softmax(1), negative sampling(0)         | 0       |
| -m            | min_count of a word to be considered"                                     | 5       | 
| -n            | negative-sampling: how many 'noise words' should be drawn                 | 5       |
| -s            | size: dimensionality of word vectors (usually 100-300)                    | 100     |
| -w            | windowsize:maximum distance between current and predicted word within a sentence | 5   |

additional for fasttext:

| argument      |description                                | default |
| ------------- |-------------------------------------------|:-------:| 
| -minn         | min length of ngrams                      | 3       | 
| -maxn         | max length of ngrams                      | 6       |
| -b            | number of buckets used for hashing ngrams | 2000000 |


## 5. Query trained model

### 5.1. similarity
get 10 most similar words for input word


```shell
python3 similarity.py <PATH/TO/MODEL/>
```

### 5.2. analogy
get solution for equations like "koenig-mann+frau"
```shell
python3 analogy.py <PATH/TO/MODEL/>
```

## Visualization
make plots of word embeddings with tsne  
will be stored at ./visualizations/

```shell
python3 visualization.py
```

# Data
## DTA from 1800 to 1899 (Deutsches Textarchiv)
#### Download and unzip DTA corpus (1800-1899, Version 14.01.2020)
data will be stored at './dta_corpus/corpus/dta_xml'

```shell
wget -i get_dta_corpus.txt -P corpus && unzip ./corpus/dta_kernkorpus_1800-1899_2020-01-14.zip -d ./corpus/dta_xml
```


#### Transform DTA XML files into plain txt (with using xml2txt.xsl)
will be stored at './dta_corpus/corpus/input_txt' 

```shell
(if necessary: chmod +x ./dta2txt.sh && ./dta2txt.sh)
./dta2txt.sh
```

You can now use './dta_corpus/corpus/input_txt/ as \<PATH TO INPUT DIR\> for training
## ANNO Music Corpus
execute the scraper in ./anno_corpus
it will create one file per journal in ./anno_corpus/corpus/input_txt/

```shell
python3 anno_scraper.py
```

You can now use './anno_corpus/corpus/input_txt/' as \<PATH TO INPUT DIR\>  for training.

Work in progress:  
FastText  
save preprocessed corpus with metadata (parser arguments)  
save models with metadata (parser arguments)

# Frequency lists
Count words in trainingdata   
file content:   
count_of_word_1 **\t** word1  
count_of_word2 **\t** word2  
...

original corpus:  
stored at ' ./corpus/input_txt_freq.txt'  
includes: lowercasing, converting 'ſ' to 's', deleting punctuation

```shell
cat ./corpus/input_txt/* | tr '[:upper:]' '[:lower:]' | tr -s 'ſ' 's' | sed -e "s/[[:punct:]]\+//g" | tr -s ' ' '\n' | sort | uniq -c | sort -rn > ./corpus/input_txt_freq.txt
```

preprocessed corpus:  
count words in preprocessed data
stored at './corpus/prep_txt_freq.txt'  

```shell
cat ./corpus/prep_txt/<NUMBER_OF_PREP_CORPUS>/data/* | tr -s ' ' '\n' | sort | uniq -c | sort -rn > ./corpus/prep_txt_freq.txt
```
