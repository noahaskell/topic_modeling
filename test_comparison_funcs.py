import comparison_funcs as cf
import numpy as np


def test_permute_matrix():
    D = np.array([[9, 7, 6, 4],
                  [8, 6, 2, 5],
                  [5, 1, 6, 7],
                  [3, 8, 6, 9]])
    E = np.array([[1, 6, 5, 7],
                  [6, 2, 8, 5],
                  [8, 6, 3, 9],
                  [7, 6, 9, 4]])
    F = cf.permute_matrix(D)
    check_list = [f == e for f, e in zip(F.ravel(), E.ravel())]
    assert all(check_list), "permute_matrix failed"


def test_jaccard_distance():
    set_x = set(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    set_y = set(['f', 'g', 'h', 'i', 'j'])
    jd = cf.jaccard_distance(set_x, set_y)
    assert jd == 0.8, "jaccard distance calculated incorrectly"


def test_model_diff():
    from gensim import models, corpora
    with open('sampledata.csv', 'r') as f:
        docs_full = f.readlines()
    docs = [x.split(',')[2].split() for x in docs_full[1:]]
    for doc in docs:
        while '<NUMBER>' in doc:
            doc.remove('<NUMBER>')
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
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
