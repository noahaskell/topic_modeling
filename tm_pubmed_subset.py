from gensim import models, corpora


def prep_corpus(fname):
    "reads in file, makes a gensim-usable corpus out of it"

    with open(fname, 'r') as f:
        docs_full = f.readlines()

    docs = [x.split('\t')[1].split() for x in docs_full]
    for doc in docs:
        while '<NUMBER>' in doc:
            doc.remove('<NUMBER>')

    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return corpus, dictionary, docs


if __name__ == "__main__":
    corpus, dictionary, docs = prep_corpus('pubmed_subset_05.tsv')
    lda_model_a = models.LdaModel(
        corpus=corpus,
        num_topics=10,
        id2word=dictionary,
        random_state=5,
        iterations=50,
        passes=1
    )
    lda_model_b = models.LdaModel(
        corpus=corpus,
        num_topics=10,
        id2word=dictionary,
        random_state=5,  # 10
        iterations=50,
        passes=2
    )
