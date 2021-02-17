import comparison_funcs as cf
import numpy as np


def test_prep_corpus():
    scorp, sdict, sdocs = cf.prep_corpus('sampledata.tsv')
    sc_b = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    sd_b = ['Human', 'exportin-1', 'XPO1', 'key', 'nuclear-cytoplasmic']
    check_list = [x == y for x, y in zip(scorp[0], sc_b)]
    assert all(check_list), "corpus prepared incorrectly, at least in part"
    check_list = [x == y for x, y in zip(sdocs[0], sd_b)]
    assert all(check_list), "docs prepared incorrectly, at least in part"


def test_compare_models():
    D = cf.compare_models('sampledata.tsv', seeds=(0, 0))
    check_list = [x == 0 for x in D.diagonal()]
    assert all(check_list), "model comparison has gone badly awry"


def test_permute_matrix():
    D = np.array([[9, 7, 6, 4],
                  [8, 6, 2, 5],
                  [5, 1, 6, 7],
                  [3, 8, 6, 9]])
    E = np.array([[1, 6, 5, 7],
                  [6, 2, 8, 5],
                  [8, 6, 3, 9],
                  [7, 6, 9, 4]])
    F, r, c = cf.permute_matrix(D, only_matrix=False)
    check_list = [f == e for f, e in zip(F.ravel(), E.ravel())]
    assert all(check_list), "permute_matrix failed"
    q = [2, 1, 3, 0]
    b = [1, 2, 0, 3]
    qr_list = [i == j for i, j in zip(q, r)]
    bc_list = [i == j for i, j in zip(b, c)]
    assert all(qr_list), "permuted row indices incorrect"
    assert all(bc_list), "permuted col indices incorrect"


def test_jaccard_distance():
    set_x = set(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    set_y = set(['f', 'g', 'h', 'i', 'j'])
    jd = cf.jaccard_distance(set_x, set_y)
    assert jd == 0.8, "jaccard distance calculated incorrectly"


def test_model_diff():
    from gensim import models
    corpus, dictionary, docs = cf.prep_corpus('sampledata.csv', delim=',')
    mod_a = models.LdaModel(
        corpus=corpus,
        num_topics=5,
        id2word=dictionary,
        random_state=0
    )
    mod_b = models.LdaModel(
        corpus=corpus,
        num_topics=5,
        id2word=dictionary,
        random_state=1
    )
    D1 = mod_a.diff(mod_b,
                    distance='jaccard',
                    num_words=50,
                    annotation=False,
                    normed=False)[0]
    D2 = cf.model_diff(mod_a, mod_b, n_words=50)
    check_list = [x == y for x, y in zip(D1.ravel(), D2.ravel())]
    assert all(check_list), "distance matrices unexpectedly unequal"
