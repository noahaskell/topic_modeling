import numpy as np
import requests
from gensim import models, corpora


def prep_corpus(fname, delim='\t'):
    """
    Reads in file, makes a gensim-usable corpus out of it

    :param fname: (path and) name of file with lemmatized docs, one per line
    :param delim: delimiter in corpus file (default '\t')
    :returns: corpus, dictionary, docs
               corpus: list of lists of bags-of-words (idx, count) tuples
               dictionary: gensim.corpora.Dictionary instance
               docs: list of lists of words in the bags in corpus
    """

    if fname[:4] == 'http':
        r = requests.get(fname)
        docs_full = r.text.split('\n')
        while '' in docs_full:
            docs_full.remove('')
    else:
        with open(fname, 'r') as f:
            docs_full = f.readlines()

    docs = [x.split(delim)[1].split() for x in docs_full]
    for doc in docs:
        while '<NUMBER>' in doc:
            doc.remove('<NUMBER>')

    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return corpus, dictionary, docs


def compare_models(fname, n_topic=10, seeds=(0, 1),
                   iters=(50, 50), passes=(1, 1),
                   workers=None, pr_alpha='symmetric'):
    """
    Fits two LDA models to the same corpus (processed from docs in file fname)
    setting the random_state, iterations, and passes parameters, returns
    a jaccard distance matrix of the topic differences between the models

    :param fname: (path and) name of file with lemmatized docs, one per line
                  to be passed to prep_corpus function
    :param n_topic: the number of topics for each model to use
    :param seeds: tuple with random_state values to be passed to LdaModel func
    :param iters: tuple with iterations values to be passed to LdaModel func
    :param passes: tuple with passes values to be passed to LdaModel func
    :param workers: None or int, if int use LdaMulticore w/ that many workers
    :param pr_alpha: {'symmetric', 'asymmetric'}, prior on alpha
    :returns: distance matrix from calling model_diff function below or
              mod_a.diff(mod_b, distance='jaccard', normed=False)
    """
    corpus, dictionary, docs = prep_corpus(fname)
    if workers is None:
        m_a = models.LdaModel(
            corpus=corpus,
            num_topics=n_topic,
            id2word=dictionary,
            random_state=seeds[0],
            iterations=iters[0],
            passes=passes[0],
            alpha=pr_alpha
        )
        m_b = models.LdaModel(
            corpus=corpus,
            num_topics=n_topic,
            id2word=dictionary,
            random_state=seeds[1],
            iterations=iters[1],
            passes=passes[1],
            alpha=pr_alpha
        )
    elif isinstance(workers, int):
        m_a = models.LdaMulticore(
            corpus=corpus,
            num_topics=n_topic,
            id2word=dictionary,
            random_state=seeds[0],
            iterations=iters[0],
            passes=passes[0],
            workers=workers,
            alpha=pr_alpha
        )
        m_b = models.LdaMulticore(
            corpus=corpus,
            num_topics=n_topic,
            id2word=dictionary,
            random_state=seeds[1],
            iterations=iters[1],
            passes=passes[1],
            workers=workers,
            alpha=pr_alpha
        )
    D, A = m_a.diff(m_b, distance='jaccard', normed=False)
    return D


def permute_matrix(D, only_matrix=True):
    """
    Permutes rows and columns of distance matrix to minimize distance between
    topics from two models, corresponding to the rows and columns of the matrix

    :param D: matrix with elements indicating distance between topics from two
              models with rows corresponding to topics from one model, columns
              to another
    :param only_matrix: bool, if True, only return permuted matrix, else return
                        permuted row and column indices, too
    :returns: permuted matrix (and optionally permuted row, col indices)
    """
    nr, nc = D.shape
    nidx = np.min((nr, nc))
    D = D.copy()
    ridx, cidx = list(range(nr)), list(range(nc))
    for idx in range(nidx):
        M = D[idx:, idx:].copy()
        r = M.argmin(0)
        c = np.array([M[r[i], i] for i in range(nc-idx)]).argmin()
        # swap rows
        xa = D[idx, :].copy()
        xb = D[r[c]+idx, :].copy()
        D[idx, :] = xb
        D[r[c]+idx, :] = xa
        ridx[r[c]+idx], ridx[idx] = ridx[idx], ridx[r[c]+idx]
        # swap columns
        ya = D[:, idx].copy()
        yb = D[:, c+idx].copy()
        D[:, idx] = yb
        D[:, c+idx] = ya
        cidx[c+idx], cidx[idx] = cidx[idx], cidx[c+idx]
    if only_matrix:
        return D
    else:
        return D, ridx, cidx


def jaccard_distance(set_a, set_b):
    """
    Calculates jaccard distance between two sets

    :param set_a: set of words (indices or words)
    :param set_b: set of words (indices or words)
    :returns: float, jaccard distance
    """
    set_a = set(set_a)  # just to make sure
    set_b = set(set_b)  # just to make sure
    inter = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return 1 - len(inter)/len(union)


def model_diff(model_a, model_b, n_words=100):
    """
    Calculates the pairwise jaccard distance between topics from
    model_a and model_b
     - same as matrix from model_a.diff(model_b,
                                        distance='jaccard',
                                        num_words=100,
                                        normed=False)

    :param model_a: gensim lda model
    :param model_b: gensim lda model
    :param n_words: int, number of words to use in calculating distance
    :returns: distance matrix
    """
    n_topic_a = model_a.num_topics
    n_topic_b = model_b.num_topics
    D = np.zeros((n_topic_a, n_topic_b))
    for t_a in range(n_topic_a):
        for t_b in range(n_topic_b):
            set_a = set(
                [x[0] for x in model_a.get_topic_terms(t_a, topn=n_words)]
            )
            set_b = set(
                [x[0] for x in model_b.get_topic_terms(t_b, topn=n_words)]
            )
            D[t_a, t_b] = jaccard_distance(set_a, set_b)
    return D
