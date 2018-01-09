
Table of Contents
=================

  * [Articles](#articles)
    * [2018\-01](#2018-01)
      * [Unsupervised Low\-Dimensional Vector Representations for Words, Phrases  and Text that are Transparent, Scalable, and produce Similarity Metrics that  are Complementary to Neural Embeddings](#unsupervised-low-dimensional-vector-representations-for-words-phrases--and-text-that-are-transparent-scalable-and-produce-similarity-metrics-that--are-complementary-to-neural-embeddings)

Articles
========
## 2018-01
### Unsupervised Low-Dimensional Vector Representations for Words, Phrases  and Text that are Transparent, Scalable, and produce Similarity Metrics that  are Complementary to Neural Embeddings

**Authors:** Neil R. Smalheiser, Gary Bonifield

**Abstract:** Neural embeddings are a popular set of methods for representing words, phrases or text as a low dimensional vector (typically 50-500 dimensions). However, it is difficult to interpret these dimensions in a meaningful manner, and creating neural embeddings requires extensive training and tuning of multiple parameters and hyperparameters. We present here a simple unsupervised method for representing words, phrases or text as a low dimensional vector, in which the meaning and relative importance of dimensions is transparent to inspection. We have created a near-comprehensive vector representation of words, and selected bigrams, trigrams and abbreviations, using the set of titles and abstracts in PubMed as a corpus. This vector is used to create several novel implicit word-word and text-text similarity metrics. The implicit word-word similarity metrics correlate well with human judgement of word pair similarity and relatedness, and outperform or equal all other reported methods on a variety of biomedical benchmarks, including several implementations of neural embeddings trained on PubMed corpora. Our implicit word-word metrics capture different aspects of word-word relatedness than word2vec-based metrics and are only partially correlated (rho = ~0.5-0.8 depending on task and corpus). The vector representations of words, bigrams, trigrams, abbreviations, and PubMed title+abstracts are all publicly available from [URL](http://arrowsmith.psych.uic.edu) for release under CC-BY-NC license. Several public web query interfaces are also available at the same site, including one which allows the user to specify a given word and view its most closely related terms according to direct co-occurrence as well as different implicit similarity metrics.

**URL:** https://arxiv.org/abs/1801.01884

**Notes:** medical-related paper on word embeddings; guys used few tricks over word2vec, like weighted score for 1- & 2-grams or list of important terms; results show their embedding actually improve relatedness of terms for humans; with code!

