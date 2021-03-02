import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()

df_i = pd.read_csv('df_iters_50_to_1600.csv')
df_p = pd.read_csv('df_passes_1_to_6.csv')
df_sip = pd.read_csv('df_seed_iter_pass_mc.csv')

# same seeds, different n_topic, iterations
x_lab_i = '# topics\nmin(iterations)\nmax(iterations)'
df_i[x_lab_i] = df_i['n_topic'].astype(str) + '\n' + \
              df_i['iter_a'].astype(str) + '\n' + df_i['iter_b'].astype(str)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.violinplot(x=x_lab_i, y='min', cut=0, data=df_i, ax=ax)
ax.set(ylabel='min(distance)',
       title='min(dist) x iterations',
       ylim=(0, .2))
fig.savefig('tm_min_dist_ntop_iter.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.violinplot(x=x_lab_i, y='mad', cut=0, data=df_i, ax=ax)
ax.set(ylabel='median abs deviation',
       title='mad x iterations',
       ylim=(0.02, .15))
fig.savefig('tm_mad_dist_ntop_iter.png', bbox_inches='tight')
plt.close(fig)

# same seeds, different n_topic, passes
x_lab_p = '# topics\nmin(passes)\nmax(passes)'
df_p[x_lab_p] = df_p['n_topic'].astype(str) + '\n' + \
              df_p['passes_a'].astype(str) + '\n' \
              + df_p['passes_b'].astype(str)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.violinplot(x=x_lab_p, y='min', cut=0, data=df_p, ax=ax)
ax.set(ylabel='min(distance)',
       title='min(dist) x passes',
       ylim=(0, .2))
fig.savefig('tm_min_dist_ntop_pass.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.violinplot(x=x_lab_p, y='mad', cut=0, data=df_p, ax=ax)
ax.set(ylabel='median abs deviation',
       title='mad x passes',
       ylim=(0.02, .15))
fig.savefig('tm_mad_dist_ntop_pass.png', bbox_inches='tight')
plt.close(fig)

# different seeds, n_topic, iterations, passes
df_sip[x_lab_i] = df_sip['n_topic'].astype(str) + '\n' + \
                  df_sip['iter_a'].astype(str) + '\n' + \
                  df_sip['iter_b'].astype(str)

df_sip[x_lab_p] = df_sip['n_topic'].astype(str) + '\n' + \
                  df_sip['passes_a'].astype(str) + '\n' + \
                  df_sip['passes_b'].astype(str)

fig, ax = plt.subplots(1, 1, figsize=(18, 9))
sns.violinplot(x=x_lab_i, y='min', cut=0, data=df_sip, ax=ax)
ax.set(ylabel='min(distance)',
       title='min(dist) x iters',
       ylim=(0, 1))
fig.savefig('tm_min_dist_ntop_iter_seeds.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(18, 9))
sns.violinplot(x=x_lab_p, y='min', cut=0, data=df_sip, ax=ax)
ax.set(ylabel='min(distance)',
       title='min(dist) x passes',
       ylim=(0, 1))
fig.savefig('tm_min_dist_ntop_pass_seeds.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(18, 9))
sns.violinplot(x=x_lab_i, y='mad', cut=0, data=df_sip, ax=ax)
ax.set(ylabel='mad',
       title='mad x iters',
       ylim=(0.02, .15))
fig.savefig('tm_mad_dist_ntop_iter_seeds.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(18, 9))
sns.violinplot(x=x_lab_p, y='mad', cut=0, data=df_sip, ax=ax)
ax.set(ylabel='mad',
       title='mad x passes',
       ylim=(0.02, .15))
fig.savefig('tm_mad_dist_ntop_pass_seeds.png', bbox_inches='tight')
plt.close(fig)
