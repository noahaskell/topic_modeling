# Stability of Topics in LDA Topic Modeling

As part of a big project that uses [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) (LDA) topic models, we are working on investigating the stability of the topics that these models "discover".

This repo contains scripts and data sets for exploring (a) sources of variation in the topics found by LDA topic models and (b) visualization of these models and this variation.

The bigger project uses [MALLET](http://mallet.cs.umass.edu), but for the purposes of these investigations, we're using [Gensim](https://radimrehurek.com/gensim/), specifically `gensim.models.LdaModel` and `gensim.models.LdaMulticore`.

Some initial exploration found that, for a given, fixed corpus, the parameters `random_state`, `iterations`, and `passes` all influence the stability of topics found. For example, if we fit two models to the same corpus, using the same `random_state` and `passes` parameters for each, but given each a different value for `iterations`, the two models tend to find slightly different topics.

Here is a heatmap showing the same for two models with `random_state=5`, `iterations=50`, and `passes=1` (i.e., identical models):

<img alt="Jaccard distance between topics for identical models" src="https://github.com/noahaskell/topic_modeling/blob/main/figures/id_heatmap.png" width="60%"/>

Here is a heatmap showing (non-normed) [Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) between topics for two models fit with `random_state=5`, `passes=1` and `iterations` equal to `50` for one model (A), `100` for the other (B):

<img alt="Jaccard distance between topics for models with different iterations" src="https://github.com/noahaskell/topic_modeling/blob/main/figures/it_heatmap.png" width="60%"/>

Here is a heatmap showing the same for two models with `random_state=5`, `iterations=50`, and `passes` equal to `1` (A) or `2` (B):

<img alt="Jaccard distance between topics for models with different passes" src="https://github.com/noahaskell/topic_modeling/blob/main/figures/ps_heatmap.png" width="60%"/>

Here is a heatmap showing the same for two models with `random_state` equal to `5` (A) or `10` (B), `iterations=50`, and `passes=1`:

<img alt="Jaccard distance between topics for models with different random states" src="https://github.com/noahaskell/topic_modeling/blob/main/figures/rs_heatmap.png" width="60%"/>

These heatmaps illustrate how differences in `iterations` and/or `passes` can produce some variation in the topics across models, but that differences in `random_state` produce *very* different topics across two models (at least with `iterations=50` and `passes=1`).

In order to systematically investigate this, we fit a bunch of pairs of models, varying `n_topic`, `iterations`, `passes`, and `random_state`, either in isolation (i.e., with other parameters held constant) or simultaneously, and calculating a number of statistics describing the (Jaccard) distances between topics (min, max, mean, median, std, mad). Of particular interest is the minimum distance between the topics found by two models, though this statistic has important limitations.

Specifically, if the minimum is (near) zero, then the two models have at least one topic that is (nearly) the same. On the other hand, if the minimum is not near zero, then the topics in the two models are not the same. That is, a minimum distance near zero is a necessary but not sufficient condition for two models to have the same topics.

Here is a figure illustrating how the number of `iterations` affects the minimum distance, holding `passes` constant at `1` and giving each model the same `random_state` value (a different value for each of 50 repetitions):

<img alt="Minimum Jaccard distance between topics for models with different iterations" src="https://github.com/noahaskell/topic_modeling/blob/main/figures/tm_min_dist_ntop_iter.png" width="100%"/>

(more soon...)
