
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
      * [Tensor Comprehensions: Framework\-Agnostic High\-Performance Machine  Learning Abstractions](#tensor-comprehensions-framework-agnostic-high-performance-machine--learning-abstractions)
      * [Ranking Sentences for Extractive Summarization with Reinforcement  Learning](#ranking-sentences-for-extractive-summarization-with-reinforcement--learning)
      * [Deep contextualized word representations](#deep-contextualized-word-representations)
      * [Latent Topic Conversational Models](#latent-topic-conversational-models)
    * [2018\-03](#2018-03)
      * [Achieving Human Parity on AutomaticChinese to English News Translation](#achieving-human-parity-on-automaticchinese-to-english-news-translation)
      * [Fast Decoding in Sequence Models using Discrete Latent Variables](#fast-decoding-in-sequence-models-using-discrete-latent-variables)
      * [An Analysis of Neural Language Modeling at Multiple Scales](#an-analysis-of-neural-language-modeling-at-multiple-scales)
      * [Disentangling Aspect and Opinion Words in Target\-based Sentiment  Analysis using Lifelong Learning](#disentangling-aspect-and-opinion-words-in-target-based-sentiment--analysis-using-lifelong-learning)
      * [Simple random search provides a competitive approach to reinforcement  learning](#simple-random-search-provides-a-competitive-approach-to-reinforcement--learning)
    * [2018\-04](#2018-04)
      * [Large scale distributed neural network training through online  distillation](#large-scale-distributed-neural-network-training-through-online--distillation)
      * [Frustratingly Easy Meta\-Embedding \-\- Computing Meta\-Embeddings by  Averaging Source Word Embeddings](#frustratingly-easy-meta-embedding----computing-meta-embeddings-by--averaging-source-word-embeddings)
      * [The Best of Both Worlds: Combining Recent Advances in Neural Machine  Translation](#the-best-of-both-worlds-combining-recent-advances-in-neural-machine--translation)
    * [2018\-05](#2018-05)
      * [Paper Abstract Writing through Editing Mechanism](#paper-abstract-writing-through-editing-mechanism)
      * [Zero\-Shot Dual Machine Translation](#zero-shot-dual-machine-translation)
    * [2018\-06](#2018-06)
      * [TextWorld: A Learning Environment for Text\-based Games](#textworld-a-learning-environment-for-text-based-games)

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

### Tensor Comprehensions: Framework-Agnostic High-Performance Machine  Learning Abstractions

**Authors:** Nicolas Vasilache, Oleksandr Zinenko, Theodoros Theodoridis, Priya Goyal, Zachary DeVito, William S. Moses, Sven Verdoolaege, Andrew Adams, Albert Cohen

**Abstract:** Deep learning models with convolutional and recurrent networks are now ubiquitous and analyze massive amounts of audio, image, video, text and graph data, with applications in automatic translation, speech-to-text, scene understanding, ranking user preferences, ad placement, etc. Competing frameworks for building these networks such as TensorFlow, Chainer, CNTK, Torch/PyTorch, Caffe1/2, MXNet and Theano, explore different tradeoffs between usability and expressiveness, research or production orientation and supported hardware. They operate on a DAG of computational operators, wrapping high-performance libraries such as CUDNN (for NVIDIA GPUs) or NNPACK (for various CPUs), and automate memory allocation, synchronization, distribution. Custom operators are needed where the computation does not fit existing high-performance library calls, usually at a high engineering cost. This is frequently required when new operators are invented by researchers: such operators suffer a severe performance penalty, which limits the pace of innovation. Furthermore, even if there is an existing runtime call these frameworks can use, it often doesn't offer optimal performance for a user's particular network architecture and dataset, missing optimizations between operators as well as optimizations that can be done knowing the size and shape of data. Our contributions include (1) a language close to the mathematics of deep learning called Tensor Comprehensions offering both imperative and declarative styles, (2) a polyhedral Just-In-Time compiler to convert a mathematical description of a deep learning DAG into a CUDA kernel with delegated memory management and synchronization, also providing optimizations such as operator fusion and specialization for specific sizes, (3) a compilation cache populated by an autotuner. [Abstract cutoff]

**URL:** https://arxiv.org/abs/1802.04730

**Notes:** really hot engineering work from Facebook: DSL which is really close to mathematic notation, so a researcher could write in it directly, from this DSL an algorithm generates code in CUDA a few times, the best generated code is used for production

### Ranking Sentences for Extractive Summarization with Reinforcement  Learning

**Authors:** Shashi Narayan, Shay B. Cohen, Mirella Lapata

**Abstract:** Single document summarization is the task of producing a shorter version of a document while preserving its principal information content. In this paper we conceptualize extractive summarization as a sentence ranking task and propose a novel training algorithm which globally optimizes the ROUGE evaluation metric through a reinforcement learning objective. We use our algorithm to train a neural summarization model on the CNN and DailyMail datasets and demonstrate experimentally that it outperforms state-of-the-art extractive and abstractive systems when evaluated automatically and by humans.

**URL:** https://arxiv.org/abs/1802.08636

**Notes:** new SotA in summarization; CNN for feature extraction from sentences; LSTM for document embedding; embedding based ranking of sentences which are used for summarization; ranking trained by REINFORCE; beam search analog for RL training

### Deep contextualized word representations

**Authors:** Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer

**Abstract:** We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.

**URL:** https://arxiv.org/abs/1802.05365

**Notes:** new SotA in NER & other tasks for embeddings; ELMo - n layers of bidirectional language model (4096 LSTM units each direction) and attn over layers, i.e embedding from each layer for each token is taken with softmax weight and summarized

### Latent Topic Conversational Models

**Authors:** Tsung-Hsien Wen, Minh-Thang Luong

**Abstract:** Despite much success in many large-scale language tasks, sequence-to-sequence (seq2seq) models have not been an ideal choice for conversational modeling as they tend to generate generic and repetitive responses. In this paper, we propose a Latent Topic Conversational Model (LTCM) that augments the seq2seq model with a neural topic component to better model human-human conversations. The neural topic component encodes information from the source sentence to build a global “topic” distribution over words, which is then consulted by the seq2seq model to improve generation at each time step. The experimental results show that the proposed LTCM can generate more diverse and interesting responses by sampling from its learnt latent representations. In a subjective human evaluation, the judges also confirm that LTCM is the preferred option comparing to competitive baseline models.

**URL:** https://openreview.net/forum?id=S1GUgxgCW

**Notes:** generating utterances in dialogues with VAE and topic modelling: before generating a sentense we draw a topic proportion and generate phrase according to it

### Disentangling Aspect and Opinion Words in Target-based Sentiment  Analysis using Lifelong Learning

**Authors:** Shuai Wang, Mianwei Zhou, Sahisnu Mazumder, Bing Liu, Yi Chang

**Abstract:** Given a target name, which can be a product aspect or entity, identifying its aspect words and opinion words in a given corpus is a fine-grained task in target-based sentiment analysis (TSA). This task is challenging, especially when we have no labeled data and we want to perform it for any given domain. To address it, we propose a general two-stage approach. Stage one extracts/groups the target-related words (call t-words) for a given target. This is relatively easy as we can apply an existing semantics-based learning technique. Stage two separates the aspect and opinion words from the grouped t-words, which is challenging because we often do not have enough word-level aspect and opinion labels. In this work, we formulate this problem in a PU learning setting and incorporate the idea of lifelong learning to solve it. Experimental results show the effectiveness of our approach.

**URL:** https://arxiv.org/abs/1802.05818

**Notes:** Authors are proposing nouvelle technique to extract opinion words from general vocabulary words. They use Amazon review dataset. The key idea here is to found opinion words as being close in multiple domains to base lexicon.

## 2018-03
### Simple random search provides a competitive approach to reinforcement  learning

**Authors:** Horia Mania, Aurelia Guy, Benjamin Recht

**Abstract:** A common belief in model-free reinforcement learning is that methods based on random search in the parameter space of policies exhibit significantly worse sample complexity than those that explore the space of actions. We dispel such beliefs by introducing a random search method for training static, linear policies for continuous control problems, matching state-of-the-art sample efficiency on the benchmark MuJoCo locomotion tasks. Our method also finds a nearly optimal controller for a challenging instance of the Linear Quadratic Regulator, a classical problem in control theory, when the dynamics are not known. Computationally, our random search algorithm is at least 15 times more efficient than the fastest competing model-free methods on these benchmarks. We take advantage of this computational efficiency to evaluate the performance of our method over hundreds of random seeds and many different hyperparameter configurations for each benchmark task. Our simulations highlight a high variability in performance in these benchmark tasks, suggesting that commonly used estimations of sample efficiency do not adequately evaluate the performance of RL algorithms.

**URL:** https://arxiv.org/abs/1803.07055

**Notes:** random search with a few optimizations achieve SotA on continous control tasks! essentially is is random search with scaling of update by reward std. deviation; much more sample efficient then Evolution Strategy; with code!

### Achieving Human Parity on AutomaticChinese to English News Translation

**Authors:** Hany Hassan, Anthony Aue, Chang Chen, Vishal Chowdhary, Jonathan Clark, Christian Federmann, Xuedong Huang, Marcin Junczys-Dowmunt, William Lewis, Mu Li, Shujie Liu, Tie-Yan Liu, Renqian Luo, Arul Menezes, Tao Qin, Frank Seide, Xu Tan, Fei Tian, Lijun Wu, Shuangzhi Wu, Yingce Xia, Dongdong Zhang, Zhirui Zhang, and Ming Zhou

**Abstract:** Machine translation has made rapid advances in recent years. Millions of people are using it today in online translation systems and mobile applications in order to communicate across language barriers. The question naturally arises whether such systems can approach or achieve parity with human translations. In this paper, we first address the problem of how to define and accurately measure human parity in translation. We then describe Microsoft’s machine translation system and measure the quality of its translations on the widely used WMT 2017 news translation task from Chinese to English. We find that our latest neural machine translation system has reached a new state-of-the-art, and that the translation quality is at human parity when compared to professional human translations. We also find that it significantly exceeds the quality of crowd-sourced non-professional translations.

**URL:** https://www.microsoft.com/en-us/research/uploads/prod/2018/03/final-achieving-human.pdf

**Notes:** human level performance in MT from MSFT! really strong claim proved in reliefingly narrow domain of English/Chinese news; not sure if the reference-matching measures are good for MT; arch is Transformer with joint unservised training on monolingual corpora

### Fast Decoding in Sequence Models using Discrete Latent Variables

**Authors:** Łukasz Kaiser, Aurko Roy, Ashish Vaswani, Niki Parmar, Samy Bengio, Jakob Uszkoreit, Noam Shazeer

**Abstract:** Autoregressive sequence models based on deep neural networks, such as RNNs, Wavenet and the Transformer attain state-of-the-art results on many tasks. However, they are difficult to parallelize and are thus slow at processing long sequences. RNNs lack parallelism both during training and decoding, while architectures like WaveNet and Transformer are much more parallelizable during training, yet still operate sequentially during decoding. Inspired by [arxiv:[URL](/abs/1711.00937)], we present a method to extend sequence models using discrete latent variables that makes decoding much more parallelizable. We first auto-encode the target sequence into a shorter sequence of discrete latent variables, which at inference time is generated autoregressively, and finally decode the output sequence from this shorter latent sequence in parallel. To this end, we introduce a novel method for constructing a sequence of discrete latent variables and compare it with previously introduced methods. Finally, we evaluate our model end-to-end on the task of neural machine translation, where it is an order of magnitude faster at decoding than comparable autoregressive models. While lower in BLEU than purely autoregressive models, our model achieves higher scores than previously proposed non-autogregressive translation models.

**URL:** https://arxiv.org/abs/1803.03382

**Notes:** great work from Google; like Non-Autoregressive Trasformer; vector quantization as projection on subspaces; the reconstruction loss for latent space and input with AE; convolutions instead of fully-connecteds; latent variables instead of fertility rate in NAT

### An Analysis of Neural Language Modeling at Multiple Scales

**Authors:** Stephen Merity, Nitish Shirish Keskar, Richard Socher

**Abstract:** Many of the leading approaches in language modeling introduce novel, complex and specialized architectures. We take existing state-of-the-art word level language models based on LSTMs and QRNNs and extend them to both larger vocabularies as well as character-level granularity. When properly tuned, LSTMs and QRNNs achieve state-of-the-art results on character-level (Penn Treebank, enwik8) and word-level (WikiText-103) datasets, respectively. Results are obtained in only 12 hours (WikiText-103) to 2 days (enwik8) using a single modern GPU.

**URL:** https://arxiv.org/abs/1803.08240

**Notes:** new paper from Salesforce: larger well-tuned LSTM (and QRNN) are giving SotA results in language modelling; in addition they use longer BPTT (150) and tied adaptive softmax; interestingly they found that QRNN need to be deeper than LSTM for these tasks

## 2018-04
### Large scale distributed neural network training through online  distillation

**Authors:** Rohan Anil, Gabriel Pereyra, Alexandre Passos, Robert Ormandi, George E. Dahl, Geoffrey E. Hinton

**Abstract:** Techniques such as ensembling and distillation promise model quality improvements when paired with almost any base model. However, due to increased test-time cost (for ensembles) and increased complexity of the training pipeline (for distillation), these techniques are challenging to use in industrial settings. In this paper we explore a variant of distillation which is relatively straightforward to use as it does not require a complicated multi-stage setup or many new hyperparameters. Our first claim is that online distillation enables us to use extra parallelism to fit very large datasets about twice as fast. Crucially, we can still speed up training even after we have already reached the point at which additional parallelism provides no benefit for synchronous or asynchronous stochastic gradient descent. Two neural networks trained on disjoint subsets of the data can share knowledge by encouraging each model to agree with the predictions the other model would have made. These predictions can come from a stale version of the other model so they can be safely computed using weights that only rarely get transmitted. Our second claim is that online distillation is a cost-effective way to make the exact predictions of a model dramatically more reproducible. We support our claims using experiments on the Criteo Display Ad Challenge dataset, ImageNet, and the largest to-date dataset used for neural language modeling, containing $6\times 10^{11}$ tokens and based on the Common Crawl repository of web data.

**URL:** https://arxiv.org/abs/1804.03235

**Notes:** Google large scale language modelling. It is 20 _Terabytes_ of texts with 673 billion tokens. The algorithm they propose called codistillation: distributed learning of similar models with additional loss for each model to predist close to others results.

### Frustratingly Easy Meta-Embedding -- Computing Meta-Embeddings by  Averaging Source Word Embeddings

**Authors:** Joshua Coates, Danushka Bollegala

**Abstract:** Creating accurate meta-embeddings from pre-trained source embeddings has received attention lately. Methods based on global and locally-linear transformation and concatenation have shown to produce accurate meta-embeddings. In this paper, we show that the arithmetic mean of two distinct word embedding sets yields a performant meta-embedding that is comparable or better than more complex meta-embedding learning methods. The result seems counter-intuitive given that vector spaces in different source embeddings are not comparable and cannot be simply averaged. We give insight into why averaging can still produce accurate meta-embedding despite the incomparability of the source vector spaces.

**URL:** https://arxiv.org/abs/1804.05262

**Notes:** really simple claim - averaging of different embedding makes them better; it is proved on datasets for semantic properties of words; interestingly averaging is slighly worse then concatenation but don't requires extra storage space

### The Best of Both Worlds: Combining Recent Advances in Neural Machine  Translation

**Authors:** Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Niki Parmar, Mike Schuster, Zhifeng Chen, Yonghui Wu, Macduff Hughes

**Abstract:** The past year has witnessed rapid advances in sequence-to-sequence (seq2seq) modeling for Machine Translation (MT). The classic RNN-based approaches to MT were first out-performed by the convolutional seq2seq model, which was then out-performed by the more recent Transformer model. Each of these new approaches consists of a fundamental architecture accompanied by a set of modeling and training techniques that are in principle applicable to other seq2seq architectures. In this paper, we tease apart the new architectures and their accompanying techniques in two ways. First, we identify several key modeling and training techniques, and apply them to the RNN architecture, yielding a new RNMT+ model that outperforms all of the three fundamental architectures on the benchmark WMT'14 English to French and English to German tasks. Second, we analyze the properties of each fundamental seq2seq architecture and devise new hybrid architectures intended to combine their strengths. Our hybrid models obtain further improvements, outperforming the RNMT+ model on both benchmark datasets.

**URL:** https://arxiv.org/abs/1804.09849

**Notes:** Google's paper on combination of multi-head and ol' good RNNs; it's somewhat surprisingly better in NMT task; as reguralizers authors use Label Smoothing, Dropout and Weight Decay; an ablation study shows that LS is even more important for NMT than MH attn

## 2018-05
### Paper Abstract Writing through Editing Mechanism

**Authors:** Qingyun Wang, Zhihao Zhou, Lifu Huang, Spencer Whitehead, Boliang Zhang, Heng Ji, Kevin Knight

**Abstract:** We present a paper abstract writing system based on an attentive neural sequence-to-sequence model that can take a title as input and automatically generate an abstract. We design a novel Writing-editing Network that can attend to both the title and the previously generated abstract drafts and then iteratively revise and polish the abstract. With two series of Turing tests, where the human judges are asked to distinguish the system-generated abstracts from human-written ones, our system passes Turing tests by junior domain experts at a rate up to 30% and by non-expert at a rate up to 80%.

**URL:** https://arxiv.org/abs/1805.06064

**Notes:** generate abstract from title; dataset id published; two networks: writing and editing ones, which relates to back-translation; Attentinve Revision Gate; ROUGE, METEOR and Turing tests (METEOR correlates with the latter, surprisingly)

### Zero-Shot Dual Machine Translation

**Authors:** Lierni Sestorain, Massimiliano Ciaramita, Christian Buck, Thomas Hofmann

**Abstract:** Neural Machine Translation (NMT) systems rely on large amounts of parallel data. This is a major challenge for low-resource languages. Building on recent work on unsupervised and semi-supervised methods, we present an approach that combines zero-shot and dual learning. The latter relies on reinforcement learning, to exploit the duality of the machine translation task, and requires only monolingual data for the target language pair. Experiments show that a zero-shot dual system, trained on English-French and English-Spanish, outperforms by large margins a standard NMT system in zero-shot translation performance on Spanish-French (both directions). The zero-shot dual method approaches the performance, within 2.2 BLEU points, of a comparable supervised setting. Our method can obtain improvements also on the setting where a small amount of parallel data for the zero-shot language pair is available. Adding Russian, to extend our experiments to jointly modeling 6 zero-shot translation directions, all directions improve between 4 and 15 BLEU points, again, reaching performance near that of the supervised setting.

**URL:** https://arxiv.org/abs/1805.10338

**Notes:** Google's fresh paper on semi-supervised NMT; zero-shot & back-translation inside; it is still pretty far from supervised models, but that approach adds 2-5 point to every language they tried (UN offial languages); and they opensourcing the model!

## 2018-06
### TextWorld: A Learning Environment for Text-based Games

**Authors:** Marc-Alexandre Côté, Ákos Kádár, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, Adam Trischler

**Abstract:** We introduce TextWorld, a sandbox learning environment for the training and evaluation of RL agents on text-based games. TextWorld is a Python library that handles interactive play-through of text games, as well as backend functions like state tracking and reward assignment. It comes with a curated list of games whose features and challenges we have analyzed. More significantly, it enables users to handcraft or automatically generate new games. Its generative mechanisms give precise control over the difficulty, scope, and language of constructed games, and can be used to relax challenges inherent to commercial text games like partial observability and sparse rewards. By generating sets of varied but similar games, TextWorld can also be used to study generalization and transfer learning. We cast text-based games in the Reinforcement Learning formalism, use our framework to develop a set of benchmark games, and evaluate several baseline agents on this set and the curated list.

**URL:** https://arxiv.org/abs/1806.11532

**Notes:** oh, yes! I've been waiting for that: text quests environment for RL agents! Thanks to MSR and colleagues from McGill! They use classic text adventures from the 80s as a Pierian spring, they introduce fwd and bkwd generation using generative grammars

