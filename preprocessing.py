import argparse
import nltk
import logging
import sys
import multiprocessing as mp
import string
import os
from tqdm import tqdm
import csv
from gensim.models.word2vec import PathLineSentences, LineSentence
from gensim.models.phrases import Phrases, Phraser
import spacy  # for lemmatization

"""
Preprocessing Corpus:
- lowercasing (-l)
- removing ocr mistakes (-o)
- converting ſ to s (-longs)
- lemmatization (-lemma)
- converting umlauts to digraphs (-u)
- removing punctuation (-p)
- removing numbers (-n)
- removing stopwords (-s)
- detecting bigrams and concatenate with "_" (-b)

Result:
txt files  with one preprocessed sentence per line

Usage example:
python3 preprocessing.py ./corpus_raw/ ./corpus_prep/ -l -o -longs -u -p -n -s -b
"""

parser = argparse.ArgumentParser(description='Script for preprocessing corpus for extracting word embeddings')
parser.add_argument('raw', type=str, help='path to dir with raw txt files')
parser.add_argument('target', type=str, help='target dir for storing preprocessed corpus')
parser.add_argument('-l', '--lower', action='store_true', help='convert text to lowercase')
parser.add_argument('-o', '--ocr', action='store_true', help='convert specific ocr mistakes like uͤ and oͤ to umlauts')
parser.add_argument('-longs', action='store_true', help='convert ſ to s')
# parser.add_argument('-lemma', '--lemmatization', action='store_true', help='lemmatization')
parser.add_argument('-u', '--umlauts', action='store_true', help='convert umlauts to digraphs')
parser.add_argument('-p', '--punctuation', action='store_true', help='remove punctuation')
parser.add_argument('-n', '--numbers', action='store_true', help='remove numbers')
parser.add_argument('-s', '--stopwords', action='store_true', help='remove stopwords')
parser.add_argument('-b', '--bigrams', action='store_true', help='detect and process bigrams')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='thread count')
parser.add_argument('--batch-size', type=int, default=32, help="batch size for multiprocessing")
args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def replace_umlauts(text):
    """
    Replace german umlauts and sharp s
    :param text: sentence as str
    :return: sentence as str with replaced umlauts
    """
    res = text
    res = res.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
    res = res.replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
    res = res.replace('ß', 'ss')
    return res


def replace_ocr_mistakes(text):
    """
    replaces oͤ -> ö; uͤ -> ü; aͤ -> ä  #TODO: weitere Besonderheiten?

    :param text: sentence as str
    :return: sentence as str with replaced chars
    """
    res = text
    res = res.replace('uͤ', 'ü').replace('oͤ', 'ö').replace('aͤ', 'ä')
    return res


def replace_long_s(text):
    """
    replace ſ with s
    :param text: sentence as str
    :return: sentence as str with replaced ſ
    """
    res = text
    res = res.replace('ſ', 's')
    return res


def process_text(infile, lower,
                 longs, ocr,  # lemmatization,
                 umlauts, punctuation,
                 numbers, stopwords):
    """
    processes text (parser arguments)
    :return: list of processed sentences
    """
    # read file
    text = infile.read()  # TODO: read via iterator? ('for line in file') (no nltk.sent_tokenize)

    processed_sentences = []

    # delete linebreaks
    text = text.replace('¬\n', '').strip()  # remove seperators and merge parts of word
    text = text.replace('-\n', '')
    text = text.replace('\n', ' ')  # remove linebreak for sentence recognition

    # load stopwords
    if not umlauts:
        # take original stopwords
        stop_words = nltk.corpus.stopwords.words('german')
    else:
        # convert umlauts in stopwords to digraphs
        stop_words = [replace_umlauts(word) for word in nltk.corpus.stopwords.words('german')]

    """
    if lemmatization:
        nlp = spacy.load('de_core_news_sm')  # for lemmatization
    else:
        nlp = None
    """

    # get sentences from text
    sentences = nltk.sent_tokenize(text, language='german')

    # process each sentence
    for sentence in sentences:
        if lower:
            sentence = sentence.lower()
        if ocr:
            sentence = replace_ocr_mistakes(sentence)
        if longs:
            sentence = replace_long_s(sentence)
        """
        if lemmatization:
            sentence = nlp(sentence)
            sentence = ' '.join([word.lemma_ for word in sentence])  # rechenintensiv!
        """
        if umlauts:
            sentence = replace_umlauts(sentence)
        if punctuation:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            sentence = sentence.replace('“', '')  # not in string.punctuation
            sentence = sentence.replace('„', '')  # not in string.punctuation
            sentence = sentence.replace('—', '')  # not in string.punctuation
        if numbers:
            sentence = sentence.translate(str.maketrans('', '', string.digits))
            # TODO: How to handle ²,³, ⁴, ⁵,⁶,⁷,⁸?
        if stopwords:
            words = nltk.word_tokenize(sentence)
            words = [x for x in words if x not in stop_words]
            sentence = ' '.join(words)
        if len(sentence) > 1:
            processed_sentences.append(sentence)
    return processed_sentences


def make_data_dir(path):
    path_to_prep_corpora = path
    # make output dir './dta_corpus/prep_txt/'
    if not os.path.exists(path_to_prep_corpora):
        os.makedirs(path_to_prep_corpora)

    # CORPUS_NUMBER = #corpora + 1
    corpus_number = str(len(os.listdir(path_to_prep_corpora)) + 1)

    # make directory for new preprocessed corpus
    path_to_corpus = path_to_prep_corpora + corpus_number + "/data/"
    os.makedirs(path_to_corpus)

    # path to meta_data file (save chosen args)
    meta_file = os.path.join(args.target, corpus_number, 'metadata.csv')
    return path_to_corpus, meta_file


def main():
    # output directory for preprocessed files
    data_dir, meta_file = make_data_dir(args.target)

    # read and process corpus (args.raw)
    for orig_file in tqdm(os.listdir(args.raw)):

        # create filename for preprocessed orig_file
        prep_file = data_dir + str(os.path.splitext(orig_file)[0]) + "_prep.txt"

        # read orig_file and preprocess
        with open(os.path.join(args.raw, orig_file), 'r') as inputfile:
            # sents = list of preprocessed sentences
            sents = process_text(inputfile, args.lower,
                                 args.longs,  # args.lemmatization,
                                 args.ocr, args.umlauts,
                                 args.punctuation, args.numbers,
                                 args.stopwords)

            # save in outputfile, one sentence per line
            with open(prep_file, 'w+') as outputfile:
                for s in sents:
                    outputfile.write(s + '\n')
                # logging.info('preprocessing of {} finished'.format(orig_file))

    if args.bigrams:
        sentences = PathLineSentences(data_dir)
        phrases = Phrases(sentences, min_count=5, threshold=10)
        # print(list(phrases[sentences]))

        filenames = os.listdir(data_dir)
        for file in filenames:
            path = os.path.join(data_dir, file)
            with open(path, "r+") as f:
                tokenized_sentences = phrases[LineSentence(path)]
                f.seek(0)
                for s in tokenized_sentences:
                    f.write('{}\n'.format(' '.join(s)))
                f.truncate()

    # save args to csv as metadata for preprocessed corpus
    with open(meta_file, 'w+') as metadata:
        args_dict = args.__dict__
        args_dict['corpus_dir'] = data_dir
        writer = csv.writer(metadata)
        writer.writerow(args_dict.keys())
        writer.writerow(args_dict.values())
        logging.info('params used for preprocessing saved at {:s}'.format(meta_file))

    logging.info('preprocessed files saved at {:s}'.format(data_dir))


if __name__ == "__main__":
    main()
