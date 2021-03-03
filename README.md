# Stability of Topics in LDA Topic Modeling

As part of a big project that uses [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) (LDA) topic models, we are working on investigating the stability of the topics that these models "discover".

This repo contains scripts and data sets for exploring (a) sources of variation in the topics found by LDA topic models and (b) visualization of these models and this variation.

The bigger project uses [MALLET](http://mallet.cs.umass.edu), but for the purposes of these investigations, we're using [Gensim](https://radimrehurek.com/gensim/), specifically `gensim.models.LdaModel` and `gensim.models.LdaMulticore`.

Some initial exploration found that, for a given, fixed corpus, the parameters `random_state`, `iterations`, and `passes` all influence the stability of topics found. For example, if we fit two models to the same corpus, using the same `random_state` and `passes` parameters for each, but given each a different value for `iterations`, the two models tend to find slightly different topics.

Here is a heatmap showing the same for two models with `random_state=5`, `iterations=50`, and `passes=1` (i.e., identical models):

<img alt="Jaccard distance between topics for models with different passes" src="https://github.com/noahaskell/topic_modeling/blob/main/figures/id_heatmap.png" width="60%"/>

Here is a heatmap showing (non-normed) [Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) between topics for two models fit with `random_state=5`, `passes=1` and `iterations` equal to `50` for one model (A), `100` for the other (B):

<img alt="Jaccard distance between topics for models with different iterations" src="https://github.com/noahaskell/topic_modeling/blob/main/figures/it_heatmap.png" width="60%"/>

Here is a heatmap showing the same for two models with `random_state=5`, `iterations=50`, and `passes` equal to `1` (A) or `2` (B):

<img alt="Jaccard distance between topics for models with different passes" src="https://github.com/noahaskell/topic_modeling/blob/main/figures/ps_heatmap.png" width="60%"/>

Here is a heatmap showing the same for two models with `random_state` equal to `5` (A) or `10` (B), `iterations=50`, and `passes=1`:

<img alt="Jaccard distance between topics for models with different passes" src="https://github.com/noahaskell/topic_modeling/blob/main/figures/rs_heatmap.png" width="60%"/>

