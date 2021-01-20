from gensim.models import KeyedVectors
import argparse
from pathlib import Path
import re

"""
loading stored model (from training.py)
calculating analogy for input words

"""

parser = argparse.ArgumentParser(description='script for computing similarity between words')
parser.add_argument('model', help='path to stored model')
args = parser.parse_args()


def out_of_vocab(words, vocab):
    oov = []
    for w in words:
        if w.strip() not in vocab:
            oov.append(w.strip())
    return oov


print("..loading model..")
wv_from_bin = KeyedVectors.load_word2vec_format(Path(args.model), binary=True)
while True:
    equation = input("get analogy (example input: 'koenig-mann+frau') (quit with 'q'): \n")
    if equation == "q":
        break
    else:
        all_input_words = re.split('[-+]', equation)
        # minus_context = equation.split('-')
        negative = [all_input_words[0].strip()]
        print(negative)
        positive = [word.strip() for word in all_input_words if word.strip() not in negative]
        print(positive)
        # left = 0
        # right = 1
        # negative = [minus_context[right].split('+')[left].strip()]
        # positive = [word.strip() for word in all_input_words if word not in negative]
        if len(all_input_words) > 3:
            print("too many words as input")
        elif len(positive) == 2 and len(negative) == 1:
            oov = out_of_vocab(all_input_words, wv_from_bin.vocab)
            if len(oov) >= 1:
                print("you used out of vocabulary words: " + str(oov))
            else:
                for (word, sim) in (list((wv_from_bin.most_similar(positive=positive, negative=negative)))):
                    print((word, sim))
               # print(wv_from_bin.most_similar(positive=positive, negative=negative))
        else:
            print("your equation has not the right form of 'a-b+c")
