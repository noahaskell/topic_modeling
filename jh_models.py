from gensim import models, corpora
import pyLDAvis.gensim



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


filename = 'pubmed_subset_05.tsv'

corp, dictionary, documents = prep_corpus(filename)

lda_model = models.LdaModel(
    corpus=corp,
    num_topics=10,
    id2word=dictionary,
    random_state=5,
    iterations=50,
    passes=1
)

# print(lda_model_a)

lda_visualization = pyLDAvis.gensim.prepare(lda_model, corp, dictionary, sort_topics=False)
pyLDAvis.display(lda_visualization)