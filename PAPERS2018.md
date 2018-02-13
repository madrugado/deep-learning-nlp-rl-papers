
Table of Contents
=================

  * [Articles](#articles)
    * [2018\-01](#2018-01)
      * [Unsupervised Low\-Dimensional Vector Representations for Words, Phrases  and Text that are Transparent, Scalable, and produce Similarity Metrics that  are Complementary to Neural Embeddings](#unsupervised-low-dimensional-vector-representations-for-words-phrases--and-text-that-are-transparent-scalable-and-produce-similarity-metrics-that--are-complementary-to-neural-embeddings)
      * [Knowledge\-based Word Sense Disambiguation using Topic Models](#knowledge-based-word-sense-disambiguation-using-topic-models)
      * [Unsupervised Part\-of\-Speech Induction](#unsupervised-part-of-speech-induction)
      * [MaskGAN: Better Text Generation via Filling in the \_\_\_\_\_\_](#maskgan-better-text-generation-via-filling-in-the-______)
    * [2018\-02](#2018-02)
      * [Improving Variational Encoder\-Decoders in Dialogue Generation](#improving-variational-encoder-decoders-in-dialogue-generation)
      * [TextZoo, a New Benchmark for Reconsidering Text Classification](#textzoo-a-new-benchmark-for-reconsidering-text-classification)

Articles
========
## 2018-01
### Unsupervised Low-Dimensional Vector Representations for Words, Phrases  and Text that are Transparent, Scalable, and produce Similarity Metrics that  are Complementary to Neural Embeddings

**Authors:** Neil R. Smalheiser, Gary Bonifield

**Abstract:** Neural embeddings are a popular set of methods for representing words, phrases or text as a low dimensional vector (typically 50-500 dimensions). However, it is difficult to interpret these dimensions in a meaningful manner, and creating neural embeddings requires extensive training and tuning of multiple parameters and hyperparameters. We present here a simple unsupervised method for representing words, phrases or text as a low dimensional vector, in which the meaning and relative importance of dimensions is transparent to inspection. We have created a near-comprehensive vector representation of words, and selected bigrams, trigrams and abbreviations, using the set of titles and abstracts in PubMed as a corpus. This vector is used to create several novel implicit word-word and text-text similarity metrics. The implicit word-word similarity metrics correlate well with human judgement of word pair similarity and relatedness, and outperform or equal all other reported methods on a variety of biomedical benchmarks, including several implementations of neural embeddings trained on PubMed corpora. Our implicit word-word metrics capture different aspects of word-word relatedness than word2vec-based metrics and are only partially correlated (rho = ~0.5-0.8 depending on task and corpus). The vector representations of words, bigrams, trigrams, abbreviations, and PubMed title+abstracts are all publicly available from [URL](http://arrowsmith.psych.uic.edu) for release under CC-BY-NC license. Several public web query interfaces are also available at the same site, including one which allows the user to specify a given word and view its most closely related terms according to direct co-occurrence as well as different implicit similarity metrics.

**URL:** https://arxiv.org/abs/1801.01884

**Notes:** medical-related paper on word embeddings; guys used few tricks over word2vec, like weighted score for 1- & 2-grams or list of important terms; results show their embedding actually improve relatedness of terms for humans; with code!

### Knowledge-based Word Sense Disambiguation using Topic Models

**Authors:** Devendra Singh Chaplot, Ruslan Salakhutdinov

**Abstract:** Word Sense Disambiguation is an open problem in Natural Language Processing which is particularly challenging and useful in the unsupervised setting where all the words in any given text need to be disambiguated without using any labeled data. Typically WSD systems use the sentence or a small window of words around the target word as the context for disambiguation because their computational complexity scales exponentially with the size of the context. In this paper, we leverage the formalism of topic model to design a WSD system that scales linearly with the number of words in the context. As a result, our system is able to utilize the whole document as the context for a word to be disambiguated. The proposed method is a variant of Latent Dirichlet Allocation in which the topic proportions for a document are replaced by synset proportions. We further utilize the information in the WordNet by assigning a non-uniform prior to synset distribution over words and a logistic-normal prior for document distribution over synsets. We evaluate the proposed method on Senseval-2, Senseval-3, SemEval-2007, SemEval-2013 and SemEval-2015 English All-Word WSD datasets and show that it outperforms the state-of-the-art unsupervised knowledge-based WSD system by a significant margin.

**URL:** https://arxiv.org/abs/1801.01900

**Notes:** word sense disambiguation with wordnet, assigning prior as normal distribution; the parameters of normal distribution are determined from corpus at hand; the topics are being modelled by synset disrtibution instead of word themselves

### Unsupervised Part-of-Speech Induction

**Authors:** Omid Kashefi

**Abstract:** Part-of-Speech (POS) tagging is an old and fundamental task in natural language processing. While supervised POS taggers have shown promising accuracy, it is not always feasible to use supervised methods due to lack of labeled data. In this project, we attempt to unsurprisingly induce POS tags by iteratively looking for a recurring pattern of words through a hierarchical agglomerative clustering process. Our approach shows promising results when compared to the tagging results of the state-of-the-art unsupervised POS taggers.

**URL:** https://arxiv.org/abs/1801.03564

**Notes:** unsupervised PoS-tagging; the author use classic backward-forward algorithm to cluster tags produced by HMM; it shows promicing results - only 10% worse that SotA

### MaskGAN: Better Text Generation via Filling in the ______

**Authors:** William Fedus, Ian Goodfellow, Andrew M. Dai

**Abstract:** Neural text generation models are often autoregressive language models or seq2seq models. These models generate text by sampling words sequentially, with each word conditioned on the previous word, and are state-of-the-art for several machine translation and summarization benchmarks. These benchmarks are often defined by validation perplexity even though this is not a direct measure of the quality of the generated text. Additionally, these models are typically trained via maxi- mum likelihood and teacher forcing. These methods are well-suited to optimizing perplexity but can result in poor sample quality since generating text requires conditioning on sequences of words that may have never been observed at training time. We propose to improve sample quality using Generative Adversarial Networks (GANs), which explicitly train the generator to produce high quality samples and have shown a lot of success in image generation. GANs were originally designed to output differentiable values, so discrete language generation is challenging for them. We claim that validation perplexity alone is not indicative of the quality of text generated by a model. We introduce an actor-critic conditional GAN that fills in missing text conditioned on the surrounding context. We show qualitatively and quantitatively, evidence that this produces more realistic conditional and unconditional text samples compared to a maximum likelihood trained model.

**URL:** https://arxiv.org/abs/1801.07736

**Notes:** GAN on texts which actually makes sense; gererator is standard seq2seq; discriminator has the same architecture as generator, but it has two outputs: probability for a word to be real and value function, which is used as baseline in REINFORCE for generator

## 2018-02
### Improving Variational Encoder-Decoders in Dialogue Generation

**Authors:** Xiaoyu Shen, Hui Su, Shuzi Niu, Vera Demberg

**Abstract:** Variational encoder-decoders (VEDs) have shown promising results in dialogue generation. However, the latent variable distributions are usually approximated by a much simpler model than the powerful RNN structure used for encoding and decoding, yielding the KL-vanishing problem and inconsistent training objective. In this paper, we separate the training step into two phases: The first phase learns to autoencode discrete texts into continuous embeddings, from which the second phase learns to generalize latent representations by reconstructing the encoded embedding. In this case, latent variables are sampled by transforming Gaussian noise through multi-layer perceptrons and are trained with a separate VED model, which has the potential of realizing a much more flexible distribution. We compare our model with current popular models and the experiment demonstrates substantial improvement in both metric-based and human evaluations.

**URL:** https://arxiv.org/abs/1802.02032

**Notes:** combined arch for better dialog gen: auto-encoder entangled with conditional VAE; variational HRED for CVAE; CVAE is trained with scheduled sampling; training of the whole model is resembling of GANs: AE or CVAE is freezed while the other has being trained

### TextZoo, a New Benchmark for Reconsidering Text Classification

**Authors:** Benyou Wang, Li Wang, Qikang Wei

**Abstract:** Text representation is a fundamental concern in Natural Language Processing, especially in text classification. Recently, many neural network approaches with delicate representation model (e.g. FASTTEXT, CNN, RNN and many hybrid models with attention mechanisms) claimed that they achieved state-of-art in specific text classification datasets. However, it lacks an unified benchmark to compare these models and reveals the advantage of each sub-components for various settings. We re-implement more than 20 popular text representation models for classification in more than 10 datasets. In this paper, we reconsider the text classification task in the perspective of neural network and get serval effects with analysis of the above results.

**URL:** https://arxiv.org/abs/1802.03656

**Notes:** conceptually simple paper, but the code for it is really useful: guys reimplemented SotA text classification architectures in one manner

