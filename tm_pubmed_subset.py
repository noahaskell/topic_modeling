from gensim import models, corpora
import seaborn as sns
from matplotlib import pyplot as plt


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

    prefix = 'ps_'
    if prefix == 'rs_':
        rs, it, ps = (5, 10), (50, 50), (1, 1)
    elif prefix == 'it_':
        rs, it, ps = (5, 5), (50, 100), (1, 1)
    elif prefix == 'ps_':
        rs, it, ps = (5, 5), (50, 50), (1, 2)

    lda_model_a = models.LdaModel(
        corpus=corpus,
        num_topics=10,
        id2word=dictionary,
        random_state=rs[0],
        iterations=it[0],
        passes=ps[0]
    )
    lda_model_b = models.LdaModel(
        corpus=corpus,
        num_topics=10,
        id2word=dictionary,
        random_state=rs[1],
        iterations=it[1],
        passes=ps[1]
    )

    D, A = lda_model_a.diff(lda_model_b,
                            distance='jaccard',
                            normed=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(D, vmin=0, vmax=1, ax=ax)
    ax.set(xlabel='Topic #, Model A',
           ylabel='Topic #, Model B',
           title='Pairwise Topic Jaccard Distance')
    fig = plt.gcf()
    fig.savefig('figures/' + prefix + 'heatmap.png', bbox_inches='tight')
    plt.close(fig)
