import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()

plot_iters = False
plot_passes = False
plot_sip = True

if plot_iters:
    df_i = pd.read_csv('df_iters_50_to_1600.csv')

    # same seeds, different n_topic, iterations
    x_lab_i = '# topics\nmin(iterations)\nmax(iterations)'
    a = df_i['n_topic'].astype(str)
    b = df_i['iter_a'].astype(str)
    c = df_i['iter_b'].astype(str)
    df_i[x_lab_i] = a + '\n' + b + '\n' + c

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.violinplot(x=x_lab_i, y='min', cut=0, data=df_i, ax=ax)
    ax.set(ylabel='min(distance)',
           title='min(dist) x iterations',
           ylim=(0, .2))
    fig.savefig('figures/tm_min_dist_ntop_iter.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.violinplot(x=x_lab_i, y='mad', cut=0, data=df_i, ax=ax)
    ax.set(ylabel='median abs deviation',
           title='mad x iterations',
           ylim=(0.02, .15))
    fig.savefig('figures/tm_mad_dist_ntop_iter.png', bbox_inches='tight')
    plt.close(fig)


if plot_passes:
    df_p = pd.read_csv('df_passes_1_to_6.csv')

    # same seeds, different n_topic, passes
    x_lab_p = '# topics\nmin(passes)\nmax(passes)'
    a = df_p['n_topic'].astype(str)
    b = df_p['iter_a'].astype(str)
    c = df_p['iter_b'].astype(str)
    df_p[x_lab_p] = a + '\n' + b + '\n' + c

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.violinplot(x=x_lab_p, y='min', cut=0, data=df_p, ax=ax)
    ax.set(ylabel='min(distance)',
           title='min(dist) x passes',
           ylim=(0, .2))
    fig.savefig('figures/tm_min_dist_ntop_pass.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.violinplot(x=x_lab_p, y='mad', cut=0, data=df_p, ax=ax)
    ax.set(ylabel='median abs deviation',
           title='mad x passes',
           ylim=(0.02, .15))
    fig.savefig('figures/tm_mad_dist_ntop_pass.png', bbox_inches='tight')
    plt.close(fig)

if plot_sip:
    sym_or_asym = 'sym'  # symmetric or asymmetric prior on alpha
    if sym_or_asym == 'sym':
        df_sip = pd.read_csv('df_seed_iter_pass_mc_sym.csv')
    elif sym_or_asym == 'asym':
        df_sip = pd.read_csv('df_seed_iter_pass_mc_asym.csv')

    # different seeds, n_topic, iterations, passes
    x_lab_i = '# topics\nmin(iterations)\nmax(iterations)'
    a = df_sip['n_topic'].astype(str)
    b = df_sip['iter_a'].astype(str)
    c = df_sip['iter_b'].astype(str)
    df_sip[x_lab_i] = a + '\n' + b + '\n' + c

    x_lab_p = '# topics\nmin(passes)\nmax(passes)'
    a = df_sip['n_topic'].astype(str)
    b = df_sip['passes_a'].astype(str)
    c = df_sip['passes_b'].astype(str)
    df_sip[x_lab_p] = a + '\n' + b + '\n' + c

    fig, ax = plt.subplots(1, 1, figsize=(18, 9))
    sns.violinplot(x=x_lab_i, y='min', cut=0, data=df_sip, ax=ax)
    ax.set(ylabel='min(distance)',
           title='min(dist) x iters',
           ylim=(0, 1))
    fig.savefig('figures/tm_min_dist_ntop_iter_seeds_' + sym_or_asym + '.png',
                bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(18, 9))
    sns.violinplot(x=x_lab_p, y='min', cut=0, data=df_sip, ax=ax)
    ax.set(ylabel='min(distance)',
           title='min(dist) x passes',
           ylim=(0, 1))
    fig.savefig('figures/tm_min_dist_ntop_pass_seeds_' + sym_or_asym + '.png',
                bbox_inches='tight')
    plt.close(fig)

    corpora = sorted(df_sip['corpus'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, cnax in enumerate(zip(corpora, axes)):
        cn, ax = cnax
        rows = df_sip['corpus'] == cn
        df_t = df_sip.loc[rows, :].copy()
        sns.violinplot(x=x_lab_i, y='min', cut=0, data=df_t, ax=ax)
        ax.text(0, .9, 'corpus #' + str(cn))
        if i in (0, 1):
            ax.set(ylabel='min(distance)',
                   title='min(dist) x iters',
                   ylim=(0, 1),
                   xlabel='')
        else:
            ax.set(ylabel='min(distance)',
                   ylim=(0, 1),
                   xlabel=x_lab_i)
    fig.savefig('figures/tm_min_dist_ntop_iter_2x2_' + sym_or_asym + '.png',
                bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, cnax in enumerate(zip(corpora, axes)):
        cn, ax = cnax
        rows = df_sip['corpus'] == cn
        df_t = df_sip.loc[rows, :].copy()
        sns.violinplot(x=x_lab_p, y='min', cut=0, data=df_t, ax=ax)
        ax.text(0, .9, 'corpus #' + str(cn))
        if i in (0, 1):
            ax.set(ylabel='min(distance)',
                   title='min(dist) x passes',
                   ylim=(0, 1),
                   xlabel='')
        else:
            ax.set(ylabel='min(distance)',
                   ylim=(0, 1),
                   xlabel=x_lab_p)
    fig.savefig('figures/tm_min_dist_ntop_pass_2x2_' + sym_or_asym + '.png',
                bbox_inches='tight')
    plt.close(fig)
