"""
train word embeddings on preprocessed corpus-files (= one sentence per line)
store as KeyedVectors (binary) at './models/<MODELTYPE>/'
--> no training after saving possible!

usage example:
python3 training.py word2vec ./corpus/prep_txt/1/data
"""

import gensim
from gensim.models.word2vec import PathLineSentences
import argparse
import os
import logging
import sys
import pickle
from collections import defaultdict

# configuration

# TODO: model_file mit arguments speichern
"""
main choices:
- sg: network architecture: Skipgram (1) otherwise CBOW (0)
- hs: training algorithm: hierarchical softmax (1) otherwise negative sampling (0)
- n: negative sampling: how many "noise words" should be drawn (usually betwenn 5-20)
- s: dimensionality of word vectors
- w: windowsize (for skipgram usually ~10, cbow ~5)
"""
parser = argparse.ArgumentParser(description="script for training word embeddings from preprocessed corpus")
parser.add_argument('model', type=str, default='word2vec', help="choose between 'word2vec' and 'fasttext'")
parser.add_argument('corpus', type=str, help="path to directory with preprocessed corpus-files (one sentence per line)")
parser.add_argument('target', type=str, help="path to directory for storing model")
parser.add_argument('-sg', '--skipgram', type=int, default=0, help="network architecture: SKIPGRAM (1) otherwise CBOW (0)")
parser.add_argument('-hs', '--softmax', type=int, default=0, help="hierarchical softmax (1) otherwise negative sampling (0)")
parser.add_argument('-m', '--mincount', type=int, default=5, help="min_count of a word to be considered")
parser.add_argument('-n', '--negative', type=int, default=5, help="how many 'noise words' should be drawn")
parser.add_argument('-s', '--size', type=int, default=100, help="dimensionality of word vectors (usually 100-300)")
parser.add_argument('-w', '--windowsize', type=int, default=5, help="maximum distance between current and predicted word within a sentence")
parser.add_argument('-t', '--threads', type=int, default=3, help="number of worker threads")


# additional parameters for fasttext (n-gram based word vectors):
parser.add_argument('-minn', '--min_n', type=int, default=3,
                    help="min length of char ngrams (only needed for FastText)")
parser.add_argument('-maxn', '--max_n', type=int, default=6,
                    help="max length of char ngrams (only needed for FastText)")
parser.add_argument('-b', '--bucket', type=int, default=2000000,
                    help="number of buckets used for hashing ngrams (only needed for FastText)")


args = parser.parse_args()
args.model = args.model.lower()  # accept input case insensitive

logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# exclude 'args.model' and args.corpus
model_params = {'sg': args.skipgram,
                'hs': args.softmax,
                'min_count': args.mincount,
                'negative': args.negative,
                'size': args.size,
                'window': args.windowsize,
                'workers': args.threads
                }


# make directory for trained model
def file_path(modeltype):
    """
    create path to directory for trained model (e.g. ./models/word2vec/<NUMBER_OF_MODEL>/ or ./models/fasttext/<NUMBER_OF_MODEL>/)
    create filename for current model
    :param modeltype: str ('word2vec' or 'fasttext')
    :return: path to file for saving model
    """
    path_to_models = os.path.join(args.target)
    if not os.path.exists(path_to_models):
        os.makedirs(path_to_models)

    # create filename for current model
    if modeltype == 'word2vec':  # suffix '_wv'
        path_to_file = os.path.join(path_to_models,
                                    'sg' + str(args.skipgram) + '_'+
                                    'hs' + str(args.softmax) + '_' +
                                    'm' + str(args.mincount) + '_' +
                                    'n' + str(args.negative) + '_' +
                                    's' + str(args.size) + '_' +
                                    'w' + str(args.windowsize) + '_' + 'wv.bin')
        return path_to_file
    elif modeltype == 'fasttext':  # suffix '_ft'
        path_to_file = os.path.join(path_to_models,
                                    'sg' + str(args.skipgram) + '_'+
                                    'hs' + str(args.softmax) + '_' +
                                    'm' + str(args.mincount) + '_' +
                                    'n' + str(args.negative) + '_' +
                                    's' + str(args.size) + '_' +
                                    'w' + str(args.windowsize) + '_' +
                                    'nmin' + str(args.min_n) + '_' +
                                    'nmax' + str(args.max_n) + '_' +
                                    'b' + str(args.bucket) + '_' + 'ft.bin')
        return path_to_file


def model_exists(par_file, params):
    """
    test, if there already exists a model trained on same corpus with same parameters
    :param par_file: file with stored parameters for each stored model
    :param params: current parameters
    :return: True, if model with same corpus and parameters already exists, else False
    """
    with open(par_file, 'rb') as f:
        saved_infos = pickle.load(f)
        for saved_model, saved_params in saved_infos.items():
            if saved_params == params:
                print("model already trained and saved at {:s}".format(saved_model))
                return True
        return False


# train word2vec model
if args.model == 'word2vec':

    output_file = file_path(args.model)  # file for storing model
    if os.path.exists(output_file):
        print('you already trained a model with this parameters')
        answer = input('continue training?(will overwrite existing model) y/n ')
        if answer == 'y':
            # train model
            logging.info("\n\ntraining word2vec model on {:s} ..".format(args.corpus))
            sentences = PathLineSentences(args.corpus)  # iterator over all files from directory
            model = gensim.models.Word2Vec(sentences=sentences, **model_params)
            model.wv.save_word2vec_format(output_file, binary=True)  # save model (no further training)
            logging.info("\n\nsaved model at {:s} ".format(output_file))
            sys.exit(0)
        else:
            print('quit training')
            sys.exit(1)
    logging.info("\n\ntraining word2vec model on {:s} ..".format(args.corpus))
    sentences = PathLineSentences(args.corpus)  # iterator over all files from directory
    model = gensim.models.Word2Vec(sentences=sentences, **model_params)
    model.wv.save_word2vec_format(output_file, binary=True)  # save model (no further training)
    logging.info("\n\nsaved model at {:s} ".format(output_file))


# train fasttext model
elif args.model == 'fasttext':
    # if model already trained (same corpus + parameters) --> quit
    output_file = file_path(args.model)
    if os.path.exists(output_file):
        print('you already trained a model with this parameters')
        answer = input('continue training?(will overwrite existing model) y/n ')
        if answer == 'y':
            # train model
            model = gensim.models.FastText(**model_params)  # instantiate
            model.build_vocab(sentences=PathLineSentences(args.corpus))
            total_words = model.corpus_total_words  # for managing the training rate (alpha) correctly
            model.train(sentences=PathLineSentences(args.corpus), total_words=total_words, epochs=args.iter)  # train
            model.save(output_file)  # save model
            sys.exit(0)
        else:
            print('quit training')
            sys.exit(1)
    model = gensim.models.FastText(**model_params)  # instantiate
    model.build_vocab(sentences=PathLineSentences(args.corpus))
    total_words = model.corpus_total_words  # for managing the training rate (alpha) correctly
    model.train(sentences=PathLineSentences(args.corpus), total_words=total_words, epochs=args.iter)  # train
    model.save(output_file)  # save model

# exception
else:
    print("unknown model '{:s}'; choose between 'word2vec' and 'fasttext'".format(args.model))
    sys.exit(1)
