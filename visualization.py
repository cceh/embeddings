from gensim.models import KeyedVectors
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os


def plot_embeddings(model, corpusname, woi):
    embeddings = []
    labels = []
    # words = list(model.wv.vocab)
    # words_selection = words[0:100]

    plot_words = []
    for sim_word, cosinus in model.most_similar(woi, topn=30):
        plot_words.append(sim_word)
    # add word of interest to plotted words
    plot_words.append(woi)

    for word in plot_words:
        # embeddings
        embeddings.append(model[word])
        # words
        labels.append(word)

    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=100000, method='exact', random_state=23)
    #  2 values per word:
    new_values = tsne_model.fit_transform(embeddings)

    x = []
    y = []
    # extract x and y values from new values
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    # save plots
    plt.savefig('./visualizations/' + woi + '_' + corpusname + '.pdf')
    # plt.show()
    plt.close()


# make dir for saving plots
if not os.path.exists('./visualizations/'):
    os.mkdir('./visualizations/')

# paths to models
model1 = KeyedVectors.load_word2vec_format(Path('./dta_corpus/models/sg0_hs0_m5_n5_s100_w5_wv.bin'), binary=True)
model2 = KeyedVectors.load_word2vec_format(Path('./anno_corpus/models/sg0_hs0_m5_n5_s100_w5_wv.bin'), binary=True)
model3 = KeyedVectors.load_word2vec_format(Path('./leipzig_corpus/models/sg0_hs0_m5_n5_s100_w5_wv.bin'), binary=True)

# tuples: (model (var), modelname (str)) 
models = [(model1, 'dta'), (model2, 'anno'), (model3, 'leipzig')]

words_of_interest = ['unheimlich', 'anmuthig', 'leidenschaftlich']
			#'weiblich','maennlich','pikant','kernig', 'wien', 'rot', 'blau', 'gelb', 'oesterreich', 'sitzen', 'ging'] 
			#'organisch', 'symmetrisch', 'fugirt', 'geistvoll', 'stimmungsvoll', 'huebsch', 'edel'
			# 'laut', 'leise', 'schoen', 'haesslich', 'gluecklich',
                  	# 'harmonisch', 'traurig', 'froehlich', 'kuenstlich',
                  	#'thematisch', 'ausdrucksvoll', 'ergreifend', 'wirkungsvoll', 'unangenehm']

# plot word of interest for every model
for model, name in models:
    for word in words_of_interest:
        if word in model.vocab:
            # word_of_interest = 'laut'
            plot_embeddings(model, name, word)
        else:
            print('Word ' + word + ' not in ' + name)
