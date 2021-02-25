import pandas as pd
import comparison_funcs as cf

# in scipy 1.6.0 it's median_abs_deviation
# google colab has scipy 1.4.1, so I downgraded to that
from scipy.stats import median_absolute_deviation as mad
from numpy import median

# NOTES
# - different seeds, same iters, passes
#   - increase iters, hold passes, see if seed diffs decrease
#   - increase passes, hold iters, see if seed diffs decrease
# - same seeds, passes, different iters
#   - increase iters, hold raw diff, see what happens
#     - e.g., 50-100, 250-300, 450-500, ...
#   - increase iters, hold rel diff, see what happens
#     - e.g., 50-100, 250-500, 450-900
# - same seeds, iters, different passes
#   - increase passes, hold raw diff
#     - e.g., 1-2, 3-4, 5-6
#   - increase passes, hold rel diff
#     - e.g., 1-2, 3-6, 5-10


def fit_compare(n_rep=10, n_topics=[10], seed_start=(0, 0), iters=[(50, 50)],
                passes=[(1, 1)], corpora=['pubmed_subset_05.tsv'],
                df_name='tm_model_comps.csv'):
    """
    For each (set of) value(s) of n_topics, seeds, iterations, passes
    and corpora, fits two topic models, calculates the non-normalized
    set of jaccard distances between topics in the two models, and then
    calculates and stores the following descriptive statistics:
    mean, median, min, max, sd, mad (median absolute deviation)

    :param n_rep: int -> number of repetitions for each combination of params
    :param n_topics: list of ints -> num_topics args
    :param seed_start: tuple of ints -> starting point for random_states
                       both incremented each iteration of this process
    :param iters: list of tuples of ints -> iterations args
    :param passes: list of tuples of ints -> passes args
    :param corpora: list of str -> file names of corpora
    :param df_name: file name for data frame to csv with stats
    """
    cols = ['n_topic', 'seed_a', 'seed_b', 'iter_a', 'iter_b',
            'passes_a', 'passes_b', 'mean', 'median', 'min', 'max',
            'sd', 'mad', 'corpus']
    df = pd.DataFrame(columns=cols)

    if seed_start[0] == seed_start[1]:
        seed_inc = 1
    else:
        seed_inc = max(seed_start) - min(seed_start)

    seeds = list(seed_start)

    for corpus in corpora:
        corp_num = corpus.split('.')[-2].split('_')[-1]
        for n_topic in n_topics:
            for n_iter in iters:
                for n_pass in passes:
                    for rep in range(n_rep):
                        D = cf.compare_models(
                                fname=corpus,
                                n_topic=n_topic,
                                seeds=seeds,
                                iters=n_iter,
                                passes=n_pass
                            )
                        V = D.ravel()
                        mean = V.mean()
                        medi = median(V)
                        d_min = V.min()
                        d_max = V.max()
                        d_std = V.std()
                        d_mad = mad(V)
                        dft = pd.DataFrame(
                                  {'n_topic': [n_topic],
                                   'seed_a': [seeds[0]],
                                   'seed_b': [seeds[1]],
                                   'iter_a': [n_iter[0]],
                                   'iter_b': [n_iter[1]],
                                   'passes_a': [n_pass[0]],
                                   'passes_b': [n_pass[1]],
                                   'mean': [mean],
                                   'median': [medi],
                                   'min': [d_min],
                                   'max': [d_max],
                                   'std': [d_std],
                                   'mad': [d_mad],
                                   'corpus': [corp_num]}
                              )
                        df = pd.concat((df, dft), axis=0)
                        seeds[0] += seed_inc
                        seeds[1] += seed_inc
                    df.to_csv(df_name, index=False)
