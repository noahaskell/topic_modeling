import numpy as np


def permute_matrix(D):
    """
    Permutes rows and columns of distance matrix ...
    """
    nr, nc = D.shape
    D = D.copy()
    for idx in range(nr):
        M = D[idx:, idx:].copy()
        r = M.argmin(0)
        c = np.array([M[r[i], i] for i in range(nc-idx)]).argmin()
        # swap rows
        xa = D[idx, :].copy()
        xb = D[r[c]+idx, :].copy()
        D[idx, :] = xb
        D[r[c]+idx, :] = xa
        # swap columns
        ya = D[:, idx].copy()
        yb = D[:, c+idx].copy()
        D[:, idx] = yb
        D[:, c+idx] = ya
    return D


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
     - same as matrix from model_a.diff(model_b, num_words=100, normed=False)

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
