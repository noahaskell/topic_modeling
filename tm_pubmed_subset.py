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
    corpus, dictionary, docs = prep_corpus()
    lda_model = models.LdaModel(
        corpus=corpus,
        num_topics=10,
        id2word=dictionary
    )
