
Table of Contents
=================

  * [Table of Contents](#table-of-contents)
  * [Articles](#articles)
    * [2017\-01](#2017-01)
      * [A Simple and Accurate Syntax\-Agnostic Neural Model for Dependency\-based Semantic Role Labeling](#a-simple-and-accurate-syntax-agnostic-neural-model-for-dependency-based-semantic-role-labeling)
      * [Reinforcement Learning via Recurrent Convolutional Neural Networks](#reinforcement-learning-via-recurrent-convolutional-neural-networks)
      * [Visualizing Residual Networks](#visualizing-residual-networks)
      * [Multi\-level Representations for Fine\-Grained Typing of Knowledge Base Entities](#multi-level-representations-for-fine-grained-typing-of-knowledge-base-entities)
      * [Neural Personalized Response Generation as Domain Adaptation](#neural-personalized-response-generation-as-domain-adaptation)
      * [Task\-Specific Attentive Pooling of Phrase Alignments Contributes to Sentence Matching](#task-specific-attentive-pooling-of-phrase-alignments-contributes-to-sentence-matching)
      * [Structural Attention Neural Networks for improved sentiment analysis](#structural-attention-neural-networks-for-improved-sentiment-analysis)
      * [Self\-Taught Convolutional Neural Networks for Short Text Clustering](#self-taught-convolutional-neural-networks-for-short-text-clustering)
      * [Textual Entailment with Structured Attentions and Composition](#textual-entailment-with-structured-attentions-and-composition)
      * [Unsupervised neural and Bayesian models for zero\-resource speech processing](#unsupervised-neural-and-bayesian-models-for-zero-resource-speech-processing)
      * [NIPS 2016 Tutorial: Generative Adversarial Networks](#nips-2016-tutorial-generative-adversarial-networks)
      * [Dense Associative Memory is Robust to Adversarial Inputs](#dense-associative-memory-is-robust-to-adversarial-inputs)
      * [A K\-fold Method for Baseline Estimation in Policy Gradient Algorithms](#a-k-fold-method-for-baseline-estimation-in-policy-gradient-algorithms)

Articles
========
## 2017-01
### A Simple and Accurate Syntax-Agnostic Neural Model for Dependency-based Semantic Role Labeling

**Authors:** Diego Marcheggiani, Anton Frolov, Ivan Titov

**Abstract:** We introduce a simple and accurate neural model for dependency-based semantic role labeling. Our model predicts predicate-argument dependencies relying on states of a bidirectional LSTM encoder. The semantic role labeler achieves respectable performance on English even without any kind of syntactic information and only using local inference. However, when automatically predicted part-of-speech tags are provided as input, it substantially outperforms all previous local models and approaches the best reported results on the CoNLL-2009 dataset. Syntactic parsers are unreliable on out-of-domain data, so standard (i.e. syntactically-informed) SRL models are hindered when tested in this setting. Our syntax-agnostic model appears more robust, resulting in the best reported results on the standard out-of-domain test set.

**URL:** https://arxiv.org/abs/1701.02593

**Notes:** New paper from Russian guys about PoS-tagging, could be useful in dialog tracking maybe.

### Reinforcement Learning via Recurrent Convolutional Neural Networks

**Authors:** Tanmay Shankar, Santosha K. Dwivedy, Prithwijit Guha

**Abstract:** Deep Reinforcement Learning has enabled the learning of policies for complex tasks in partially observable environments, without explicitly learning the underlying model of the tasks. While such model-free methods achieve considerable performance, they often ignore the structure of task. We present a natural representation of to Reinforcement Learning (RL) problems using Recurrent Convolutional Neural Networks (RCNNs), to better exploit this inherent structure. We define 3 such RCNNs, whose forward passes execute an efficient Value Iteration, propagate beliefs of state in partially observable environments, and choose optimal actions respectively. Backpropagating gradients through these RCNNs allows the system to explicitly learn the Transition Model and Reward Function associated with the underlying MDP, serving as an elegant alternative to classical model-based RL. We evaluate the proposed algorithms in simulation, considering a robot planning problem. We demonstrate the capability of our framework to reduce the cost of replanning, learn accurate MDP models, and finally re-plan with learnt models to achieve near-optimal policies.

**URL:** https://arxiv.org/abs/1701.02392

**Notes:** Reccurent CNN is a new trend, they had shown themselves as useful tool in NLP, now in RL, couldn't miss this one.

### Visualizing Residual Networks

**Authors:** Brian Chu, Daylen Yang, Ravi Tadinada

**Abstract:** Residual networks are the current state of the art on ImageNet. Similar work in the direction of utilizing shortcut connections has been done extremely recently with derivatives of residual networks and with highway networks. This work potentially challenges our understanding that CNNs learn layers of local features that are followed by increasingly global features. Through qualitative visualization and empirical analysis, we explore the purpose that residual skip connections serve. Our assessments show that the residual shortcut connections force layers to refine features, as expected. We also provide alternate visualizations that confirm that residual networks learn what is already intuitively known about CNNs in general.

**URL:** https://arxiv.org/abs/1701.02362

**Notes:** The heading is talking for itself, could be useful due to residuality now is useful everywhere: RNN, CNN, etc.

### Multi-level Representations for Fine-Grained Typing of Knowledge Base Entities

**Authors:** Yadollah Yaghoobzadeh, Hinrich Schütze

**Abstract:** Entities are essential elements of natural language. In this paper, we present methods for learning multi-level representations of entities on three complementary levels: character (character patterns in entity names extracted, e.g., by neural networks), word (embeddings of words in entity names) and entity (entity embeddings). We investigate state-of-the-art learning methods on each level and find large differences, e.g., for deep learning models, traditional ngram features and the subword model of fasttext (Bojanowski et al., 2016) on the character level; for word2vec (Mikolov et al., 2013) on the word level; and for the order-aware model wang2vec (Ling et al., 2015a) on the entity level. We confirm experimentally that each level of representation contributes complementary information and a joint representation of all three levels improves the existing embedding based baseline for fine-grained entity typing by a large margin. Additionally, we show that adding information from entity descriptions further improves multi-level representations of entities.

**URL:** https://arxiv.org/abs/1701.02025

**Notes:** fresh paper from Schuetze, triune of Char, Word, & Entity, seems to be the part of NLP Holy Grail

### Neural Personalized Response Generation as Domain Adaptation

**Authors:** Weinan Zhang, Ting Liu, Yifa Wang, Qingfu Zhu

**Abstract:** In this paper, we focus on the personalized response generation for conversational systems. Based on the sequence to sequence learning, especially the encoder-decoder framework, we propose a two-phase approach, namely initialization then adaptation, to model the responding style of human and then generate personalized responses. For evaluation, we propose a novel human aided method to evaluate the performance of the personalized response generation models by online real-time conversation and offline human judgement. Moreover, the lexical divergence of the responses generated by the 5 personalized models indicates that the proposed two-phase approach achieves good results on modeling the responding style of human and generating personalized responses for the conversational systems.

**URL:** https://arxiv.org/abs/1701.02073

**Notes:** personalized answer is really important part for seamless conversation, training the style from responses is a nice idea.

### Task-Specific Attentive Pooling of Phrase Alignments Contributes to Sentence Matching

**Authors:** Wenpeng Yin, Hinrich Schütze

**Abstract:** This work studies comparatively two typical sentence matching tasks: textual entailment (TE) and answer selection (AS), observing that weaker phrase alignments are more critical in TE, while stronger phrase alignments deserve more attention in AS. The key to reach this observation lies in phrase detection, phrase representation, phrase alignment, and more importantly how to connect those aligned phrases of different matching degrees with the final classifier. Prior work (i) has limitations in phrase generation and representation, or (ii) conducts alignment at word and phrase levels by handcrafted features or (iii) utilizes a single framework of alignment without considering the characteristics of specific tasks, which limits the framework's effectiveness across tasks. We propose an architecture based on Gated Recurrent Unit that supports (i) representation learning of phrases of arbitrary granularity and (ii) task-specific attentive pooling of phrase alignments between two sentences. Experimental results on TE and AS match our observation and show the effectiveness of our approach.

**URL:** https://arxiv.org/abs/1701.02149

**Notes:** attentive pooling for NLP tasks is very hot

### Structural Attention Neural Networks for improved sentiment analysis

**Authors:** Filippos Kokkinos, Alexandros Potamianos

**Abstract:** We introduce a tree-structured attention neural network for sentences and small phrases and apply it to the problem of sentiment classification. Our model expands the current recursive models by incorporating structural information around a node of a syntactic tree using both bottom-up and top-down information propagation. Also, the model utilizes structural attention to identify the most salient representations during the construction of the syntactic tree. To our knowledge, the proposed models achieve state of the art performance on the Stanford Sentiment Treebank dataset.

**URL:** https://arxiv.org/abs/1701.01811

**Notes:** one more attention type

### Self-Taught Convolutional Neural Networks for Short Text Clustering

**Authors:** Jiaming Xu, Bo Xu, Peng Wang, Suncong Zheng, Guanhua Tian, Jun Zhao, Bo Xu

**Abstract:** Short text clustering is a challenging problem due to its sparseness of text representation. Here we propose a flexible Self-Taught Convolutional neural network framework for Short Text Clustering (dubbed STC^2), which can flexibly and successfully incorporate more useful semantic features and learn non-biased deep text representation in an unsupervised manner. In our framework, the original raw text features are firstly embedded into compact binary codes by using one existing unsupervised dimensionality reduction methods. Then, word embeddings are explored and fed into convolutional neural networks to learn deep feature representations, meanwhile the output units are used to fit the pre-trained binary codes in the training process. Finally, we get the optimal clusters by employing K-means to cluster the learned representations. Extensive experimental results demonstrate that the proposed framework is effective, flexible and outperform several popular clustering methods when tested on three public short text datasets.

**URL:** https://arxiv.org/abs/1701.00185

**Notes:** unsupervised clusteding by CNNs!

### Textual Entailment with Structured Attentions and Composition

**Authors:** Kai Zhao, Liang Huang, Mingbo Ma

**Abstract:** Deep learning techniques are increasingly popular in the textual entailment task, overcoming the fragility of traditional discrete models with hard alignments and logics. In particular, the recently proposed attention models (Rockt\"aschel et al., 2015; Wang and Jiang, 2015) achieves state-of-the-art accuracy by computing soft word alignments between the premise and hypothesis sentences. However, there remains a major limitation: this line of work completely ignores syntax and recursion, which is helpful in many traditional efforts. We show that it is beneficial to extend the attention model to tree nodes between premise and hypothesis. More importantly, this subtree-level attention reveals information about entailment relation. We study the recursive composition of this subtree-level entailment relation, which can be viewed as a soft version of the Natural Logic framework (MacCartney and Manning, 2009). Experiments show that our structured attention and entailment composition model can correctly identify and infer entailment relations from the bottom up, and bring significant improvements in accuracy.

**URL:** https://arxiv.org/abs/1701.01126

**Notes:** yet another attention

### Unsupervised neural and Bayesian models for zero-resource speech processing

**Authors:** Herman Kamper

**Abstract:** In settings where only unlabelled speech data is available, zero-resource speech technology needs to be developed without transcriptions, pronunciation dictionaries, or language modelling text. There are two central problems in zero-resource speech processing: (i) finding frame-level feature representations which make it easier to discriminate between linguistic units (phones or words), and (ii) segmenting and clustering unlabelled speech into meaningful units. In this thesis, we argue that a combination of top-down and bottom-up modelling is advantageous in tackling these two problems. To address the problem of frame-level representation learning, we present the correspondence autoencoder (cAE), a neural network trained with weak top-down supervision from an unsupervised term discovery system. By combining this top-down supervision with unsupervised bottom-up initialization, the cAE yields much more discriminative features than previous approaches. We then present our unsupervised segmental Bayesian model that segments and clusters unlabelled speech into hypothesized words. By imposing a consistent top-down segmentation while also using bottom-up knowledge from detected syllable boundaries, our system outperforms several others on multi-speaker conversational English and Xitsonga speech data. Finally, we show that the clusters discovered by the segmental Bayesian model can be made less speaker- and gender-specific by using features from the cAE instead of traditional acoustic features. In summary, the different models and systems presented in this thesis show that both top-down and bottom-up modelling can improve representation learning, segmentation and clustering of unlabelled speech data.

**URL:** https://arxiv.org/abs/1701.00851

**Notes:** Unsurervised neural vs Bayesian approahces in speech processing

### NIPS 2016 Tutorial: Generative Adversarial Networks

**Authors:** Ian Goodfellow

**Abstract:** This report summarizes the tutorial presented by the author at NIPS 2016 on generative adversarial networks (GANs). The tutorial describes: (1) Why generative modeling is a topic worth studying, (2) how generative models work, and how GANs compare to other generative models, (3) the details of how GANs work, (4) research frontiers in GANs, and (5) state-of-the-art image models that combine GANs with other methods. Finally, the tutorial contains three exercises for readers to complete, and the solutions to these exercises.

**URL:** https://arxiv.org/abs/1701.00160

**Notes:** Goodfellow's tutorial couldn't hurt

### Dense Associative Memory is Robust to Adversarial Inputs

**Authors:** Dmitry Krotov, John J Hopfield

**Abstract:** Deep neural networks (DNN) trained in a supervised way suffer from two known problems. First, the minima of the objective function used in learning correspond to data points (also known as rubbish examples or fooling images) that lack semantic similarity with the training data. Second, a clean input can be changed by a small, and often imperceptible for human vision, perturbation, so that the resulting deformed input is misclassified by the network. These findings emphasize the differences between the ways DNN and humans classify patterns, and raise a question of designing learning algorithms that more accurately mimic human perception compared to the existing methods. Our paper examines these questions within the framework of Dense Associative Memory (DAM) models. These models are defined by the energy function, with higher order (higher than quadratic) interactions between the neurons. We show that in the limit when the power of the interaction vertex in the energy function is sufficiently large, these models have the following three properties. First, the minima of the objective function are free from rubbish images, so that each minimum is a semantically meaningful pattern. Second, artificial patterns poised precisely at the decision boundary look ambiguous to human subjects and share aspects of both classes that are separated by that decision boundary. Third, adversarial images constructed by models with small power of the interaction vertex, which are equivalent to DNN with rectified linear units (ReLU), fail to transfer to and fool the models with higher order interactions. This opens up a possibility to use higher order models for detecting and stopping malicious adversarial attacks. The presented results suggest that DAM with higher order energy functions are closer to human visual perception than DNN with ReLUs.

**URL:** https://arxiv.org/abs/1701.00939

**Notes:** The memory paper from *that* Hopfield

### A K-fold Method for Baseline Estimation in Policy Gradient Algorithms

**Authors:** Nithyanand Kota, Abhishek Mishra, Sunil Srinivasa, Xi (Peter) Chen, Pieter Abbeel

**Abstract:** The high variance issue in unbiased policy-gradient methods such as VPG and REINFORCE is typically mitigated by adding a baseline. However, the baseline fitting itself suffers from the underfitting or the overfitting problem. In this paper, we develop a K-fold method for baseline estimation in policy gradient algorithms. The parameter K is the baseline estimation hyperparameter that can adjust the bias-variance trade-off in the baseline estimates. We demonstrate the usefulness of our approach via two state-of-the-art policy gradient algorithms on three MuJoCo locomotive control tasks.

**URL:** https://arxiv.org/abs/1701.00867

**Notes:** Simple baseline for policy gradient

