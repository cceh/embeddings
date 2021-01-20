from gensim.models import KeyedVectors
import argparse
from pathlib import Path

"""
loading stored model (from training.py)
calculating most_similar words for input words

"""

parser = argparse.ArgumentParser(description='script for computing similarity between words')
parser.add_argument('model', help='path to stored model')
args = parser.parse_args()

print("..loading model..")
wv_from_bin = KeyedVectors.load_word2vec_format(Path(args.model), binary=True)
"""
KeyedVector: no further training possible:
hidden weights, vocabulary frequencies and the binary tree are missing
if further training necessary:
wv_from_bin = Word2Vec.load(args.model)
"""
wv_from_bin.init_sims(replace=True)  # free up memory: L2 normalize vectors and delete raw vectors

while True:
    word = input("get most similar words for (end with 'q'): \n")
    if word == "q":
        break
    elif word in wv_from_bin.vocab:
        similar_words = (wv_from_bin.most_similar(positive=word))
        print(similar_words)
        freq_doc = "./dta_corpus/dta_txt_prep_frequencies.txt"
        # TODO: print word frequencies to sim words

    else:
        print("\nthis word does not exist in this model\n")
