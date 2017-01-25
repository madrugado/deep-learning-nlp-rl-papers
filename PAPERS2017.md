
Table of Contents
=================

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
      * [Generating Long and Diverse Responses with Neural Conversation Models](#generating-long-and-diverse-responses-with-neural-conversation-models)
      * [Simplified Gating in Long Short\-term Memory (LSTM) Recurrent Neural Networks](#simplified-gating-in-long-short-term-memory-lstm-recurrent-neural-networks)
      * [Modularized Morphing of Neural Networks](#modularized-morphing-of-neural-networks)
      * [A Copy\-Augmented Sequence\-to\-Sequence Architecture Gives Good Performance on Task\-Oriented Dialogue](#a-copy-augmented-sequence-to-sequence-architecture-gives-good-performance-on-task-oriented-dialogue)
      * [Dialog Context Language Modeling with Recurrent Neural Networks](#dialog-context-language-modeling-with-recurrent-neural-networks)
      * [Neural Models for Sequence Chunking](#neural-models-for-sequence-chunking)
      * [DyNet: The Dynamic Neural Network Toolkit](#dynet-the-dynamic-neural-network-toolkit)
      * [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](#understanding-the-effective-receptive-field-in-deep-convolutional-neural-networks)
      * [Agent\-Agnostic Human\-in\-the\-Loop Reinforcement Learning](#agent-agnostic-human-in-the-loop-reinforcement-learning)
      * [Minimally Naturalistic Artificial Intelligence](#minimally-naturalistic-artificial-intelligence)
      * [Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks](#adversarial-variational-bayes-unifying-variational-autoencoders-and-generative-adversarial-networks)
      * [Adversarial Learning for Neural Dialogue Generation](#adversarial-learning-for-neural-dialogue-generation)
      * [A Multichannel Convolutional Neural Network For Cross\-language Dialog State Tracking](#a-multichannel-convolutional-neural-network-for-cross-language-dialog-state-tracking)
      * [Learning to Decode for Future Success](#learning-to-decode-for-future-success)
      * [Outrageously Large Neural Networks: The Sparsely\-Gated Mixture\-of\-Experts Layer](#outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer)
      * [Regularizing Neural Networks by Penalizing Confident Output Distributions](#regularizing-neural-networks-by-penalizing-confident-output-distributions)
      * [Discriminative Neural Topic Models](#discriminative-neural-topic-models)

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

### Generating Long and Diverse Responses with Neural Conversation Models

**Authors:** Louis Shao, Stephan Gouws, Denny Britz, Anna Goldie, Brian Strope, Ray Kurzweil

**Abstract:** Building general-purpose conversation agents is a very challenging task, but necessary on the road toward intelligent agents that can interact with humans in natural language. Neural conversation models — purely data-driven systems trained end-to-end on dialogue corpora — have shown great promise recently, yet they often produce short and generic responses. This work presents new training and decoding methods that improve the quality, coherence, and diversity of long responses generated using sequence-to-sequence models. Our approach adds self-attention to the decoder to maintain coherence in longer responses, and we propose a practical approach, called the glimpse-model, for scaling to large datasets. We introduce a stochastic beam-search algorithm with segment-by-segment reranking which lets us inject diversity earlier in the generation process. We trained on a combined data set of over 2.3B conversation messages mined from the web. In human evaluation studies, our method produces longer responses overall, with a higher proportion rated as acceptable and excellent as length increases, compared to baseline sequence-to-sequence models with explicit length-promotion. A back-off strategy produces better responses overall, in the full spectrum of lengths.

**URL:** https://arxiv.org/abs/1701.03185

**Notes:** more diversity for responces, we should look over this work, since the beam search isn't satisfying

### Simplified Gating in Long Short-term Memory (LSTM) Recurrent Neural Networks

**Authors:** Yuzhen Lu, Fathi M. Salem

**Abstract:** The standard LSTM recurrent neural networks while very powerful in long-range dependency sequence applications have highly complex structure and relatively large (adaptive) parameters. In this work, we present empirical comparison between the standard LSTM recurrent neural network architecture and three new parameter-reduced variants obtained by eliminating combinations of the input signal, bias, and hidden unit signals from individual gating signals. The experiments on two sequence datasets show that the three new variants, called simply as LSTM1, LSTM2, and LSTM3, can achieve comparable performance to the standard LSTM model with less (adaptive) parameters.

**URL:** https://arxiv.org/abs/1701.03441

**Notes:** that's a pity, I had the similar idea, you need to go fast with trying ideas this days!

### Modularized Morphing of Neural Networks

**Authors:** Tao Wei, Changhu Wang, Chang Wen Chen

**Abstract:** In this work we study the problem of network morphism, an effective learning scheme to morph a well-trained neural network to a new one with the network function completely preserved. Different from existing work where basic morphing types on the layer level were addressed, we target at the central problem of network morphism at a higher level, i.e., how a convolutional layer can be morphed into an arbitrary module of a neural network. To simplify the representation of a network, we abstract a module as a graph with blobs as vertices and convolutional layers as edges, based on which the morphing process is able to be formulated as a graph transformation problem. Two atomic morphing operations are introduced to compose the graphs, based on which modules are classified into two families, i.e., simple morphable modules and complex modules. We present practical morphing solutions for both of these two families, and prove that any reasonable module can be morphed from a single convolutional layer. Extensive experiments have been conducted based on the state-of-the-art ResNet on benchmark datasets, and the effectiveness of the proposed solution has been verified.

**URL:** https://arxiv.org/abs/1701.03281

**Notes:** modularization is a fresh idea, I cannot get morphing aside the brain damage (zeroing small weights) yet

### A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue

**Authors:** Mihail Eric, Christopher D. Manning

**Abstract:** Task-oriented dialogue focuses on conversational agents that participate in user-initiated dialogues on domain-specific topics. In contrast to chatbots, which simply seek to sustain open-ended meaningful discourse, existing task-oriented agents usually explicitly model user intent and belief states. This paper examines bypassing such an explicit representation by depending on a latent neural embedding of state and learning selective attention to dialogue history together with copying to incorporate relevant prior context. We complement recent work by showing the effectiveness of simple sequence-to-sequence neural architectures with a copy mechanism. Our model outperforms more complex memory-augmented models by 7% in per-response generation and is on par with the current state-of-the-art on DSTC2.

**URL:** https://arxiv.org/abs/1701.04024

**Notes:** Seq2Seq still has some tricks up its sleeve, copying as context is a bright idea

### Dialog Context Language Modeling with Recurrent Neural Networks

**Authors:** Bing Liu, Ian Lane

**Abstract:** In this work, we propose contextual language models that incorporate dialog level discourse information into language modeling. Previous works on contextual language model treat preceding utterances as a sequence of inputs, without considering dialog interactions. We design recurrent neural network (RNN) based contextual language models that specially track the interactions between speakers in a dialog. Experiment results on Switchboard Dialog Act Corpus show that the proposed model outperforms conventional single turn based RNN language model by 3.3% on perplexity. The proposed models also demonstrate advantageous performance over other competitive contextual language models.

**URL:** https://arxiv.org/abs/1701.04056

**Notes:** Another context incorporation with RNN

### Neural Models for Sequence Chunking

**Authors:** Feifei Zhai, Saloni Potdar, Bing Xiang, Bowen Zhou

**Abstract:** Many natural language understanding (NLU) tasks, such as shallow parsing (i.e., text chunking) and semantic slot filling, require the assignment of representative labels to the meaningful chunks in a sentence. Most of the current deep neural network (DNN) based methods consider these tasks as a sequence labeling problem, in which a word, rather than a chunk, is treated as the basic unit for labeling. These chunks are then inferred by the standard IOB (Inside-Outside-Beginning) labels. In this paper, we propose an alternative approach by investigating the use of DNN for sequence chunking, and propose three neural models so that each chunk can be treated as a complete unit for labeling. Experimental results show that the proposed neural sequence chunking models can achieve start-of-the-art performance on both the text chunking and slot filling tasks.

**URL:** https://arxiv.org/abs/1701.04027

**Notes:** Sequence chunking is common task for asian languages in the first place, but since we are going to go chars, for european ones too

### DyNet: The Dynamic Neural Network Toolkit

**Authors:** Graham Neubig, Chris Dyer, Yoav Goldberg, Austin Matthews, Waleed Ammar, Antonios Anastasopoulos, Miguel Ballesteros, David Chiang, Daniel Clothiaux, Trevor Cohn, Kevin Duh, Manaal Faruqui, Cynthia Gan, Dan Garrette, Yangfeng Ji, Lingpeng Kong, Adhiguna Kuncoro, Gaurav Kumar, Chaitanya Malaviya, Paul Michel, Yusuke Oda, Matthew Richardson, Naomi Saphra, Swabha Swayamdipta, Pengcheng Yin

**Abstract:** We describe DyNet, a toolkit for implementing neural network models based on dynamic declaration of network structure. In the static declaration strategy that is used in toolkits like Theano, CNTK, and TensorFlow, the user first defines a computation graph (a symbolic representation of the computation), and then examples are fed into an engine that executes this computation and computes its derivatives. In DyNet's dynamic declaration strategy, computation graph construction is mostly transparent, being implicitly constructed by executing procedural code that computes the network outputs, and the user is free to use different network structures for each input. Dynamic declaration thus facilitates the implementation of more complicated network architectures, and DyNet is specifically designed to allow users to implement their models in a way that is idiomatic in their preferred programming language (C++ or Python). One challenge with dynamic declaration is that because the symbolic computation graph is defined anew for every training example, its construction must have low overhead. To achieve this, DyNet has an optimized C++ backend and lightweight graph representation. Experiments show that DyNet's speeds are faster than or comparable with static declaration toolkits, and significantly faster than Chainer, another dynamic declaration toolkit. DyNet is released open-source under the Apache 2.0 license and available at this http URL

**URL:** https://arxiv.org/abs/1701.03980

**Notes:** The paper has remarkable list of authors - DeepMind, Google, IBM Watson, CMU, AI2 & MSR. And more! Very interesting initiative.

### Understanding the Effective Receptive Field in Deep Convolutional Neural Networks

**Authors:** Wenjie Luo, Yujia Li, Raquel Urtasun, Richard Zemel

**Abstract:** We study characteristics of receptive fields of units in deep convolutional networks. The receptive field size is a crucial issue in many visual tasks, as the output must respond to large enough areas in the image to capture information about large objects. We introduce the notion of an effective receptive field, and show that it both has a Gaussian distribution and only occupies a fraction of the full theoretical receptive field. We analyze the effective receptive field in several architecture designs, and the effect of nonlinear activations, dropout, sub-sampling and skip connections on it. This leads to suggestions for ways to address its tendency to be too small.

**URL:** https://arxiv.org/abs/1701.04128

**Notes:** The topic I was always curious about, shoud read it carefully, since now CNN are in rising at NLP field

### Agent-Agnostic Human-in-the-Loop Reinforcement Learning

**Authors:** David Abel, John Salvatier, Andreas Stuhlmüller, Owain Evans

**Abstract:** Providing Reinforcement Learning agents with expert advice can dramatically improve various aspects of learning. Prior work has developed teaching protocols that enable agents to learn efficiently in complex environments; many of these methods tailor the teacher's guidance to agents with a particular representation or underlying learning scheme, offering effective but specialized teaching procedures. In this work, we explore protocol programs, an agent-agnostic schema for Human-in-the-Loop Reinforcement Learning. Our goal is to incorporate the beneficial properties of a human teacher into Reinforcement Learning without making strong assumptions about the inner workings of the agent. We show how to represent existing approaches such as action pruning, reward shaping, and training in simulation as special cases of our schema and conduct preliminary experiments on simple domains.

**URL:** https://arxiv.org/abs/1701.04079

**Notes:** Next step for FAIR human in the loop approach

### Minimally Naturalistic Artificial Intelligence

**Authors:** Steven Stenberg Hansen

**Abstract:** The rapid advancement of machine learning techniques has re-energized research into general artificial intelligence. While the idea of domain-agnostic meta-learning is appealing, this emerging field must come to terms with its relationship to human cognition and the statistics and structure of the tasks humans perform. The position of this article is that only by aligning our agents' abilities and environments with those of humans do we stand a chance at developing general artificial intelligence (GAI). A broad reading of the famous 'No Free Lunch' theorem is that there is no universally optimal inductive bias or, equivalently, bias-free learning is impossible. This follows from the fact that there are an infinite number of ways to extrapolate data, any of which might be the one used by the data generating environment; an inductive bias prefers some of these extrapolations to others, which lowers performance in environments using these adversarial extrapolations. We may posit that the optimal GAI is the one that maximally exploits the statistics of its environment to create its inductive bias; accepting the fact that this agent is guaranteed to be extremely sub-optimal for some alternative environments. This trade-off appears benign when thinking about the environment as being the physical universe, as performance on any fictive universe is obviously irrelevant. But, we should expect a sharper inductive bias if we further constrain our environment. Indeed, we implicitly do so by defining GAI in terms of accomplishing that humans consider useful. One common version of this is need the for 'common-sense reasoning', which implicitly appeals to the statistics of physical universe as perceived by humans.

**URL:** https://arxiv.org/abs/1701.03868

**Notes:** Seems to be little bit too loud name, but we should check inside.

### Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks

**Authors:** Lars Mescheder, Sebastian Nowozin, Andreas Geiger

**Abstract:** Variational Autoencoders (VAEs) are expressive latent variable models that can be used to learn complex probability distributions from training data. However, the quality of the resulting model crucially relies on the expressiveness of the inference model used during training. We introduce Adversarial Variational Bayes (AVB), a technique for training Variational Autoencoders with arbitrarily expressive inference models. We achieve this by introducing an auxiliary discriminative network that allows to rephrase the maximum-likelihood-problem as a two-player game, hence establishing a principled connection between VAEs and Generative Adversarial Networks (GANs). We show that in the nonparametric limit our method yields an exact maximum-likelihood assignment for the parameters of the generative model, as well as the exact posterior distribution over the latent variables given an observation. Contrary to competing approaches which combine VAEs with GANs, our approach has a clear theoretical justification, retains most advantages of standard Variational Autoencoders and is easy to implement.

**URL:** https://arxiv.org/abs/1701.04722

**Notes:** Convergence of autoencoders & GANs, neat!

### Adversarial Learning for Neural Dialogue Generation

**Authors:** Jiwei Li, Will Monroe, Tianlin Shi, Alan Ritter, Dan Jurafsky

**Abstract:** In this paper, drawing intuition from the Turing test, we propose using adversarial training for open-domain dialogue generation: the system is trained to produce sequences that are indistinguishable from human-generated dialogue utterances. We cast the task as a reinforcement learning (RL) problem where we jointly train two systems, a generative model to produce response sequences, and a discriminator---analagous to the human evaluator in the Turing test-— to distinguish between the human-generated dialogues and the machine-generated ones. The outputs from the discriminator are then used as rewards for the generative model, pushing the system to generate dialogues that mostly resemble human dialogues. In addition to adversarial training we describe a model for adversarial {\em evaluation} that uses success in fooling an adversary as a dialogue evaluation metric, while avoiding a number of potential pitfalls. Experimental results on several metrics, including adversarial evaluation, demonstrate that the adversarially-trained system generates higher-quality responses than previous baselines.

**URL:** https://arxiv.org/abs/1701.06547

**Notes:** HOT! GAN-RL!

### A Multichannel Convolutional Neural Network For Cross-language Dialog State Tracking

**Authors:** Hongjie Shi, Takashi Ushio, Mitsuru Endo, Katsuyoshi Yamagami, Noriaki Horii

**Abstract:** The fifth Dialog State Tracking Challenge (DSTC5) introduces a new cross-language dialog state tracking scenario, where the participants are asked to build their trackers based on the English training corpus, while evaluating them with the unlabeled Chinese corpus. Although the computer-generated translations for both English and Chinese corpus are provided in the dataset, these translations contain errors and careless use of them can easily hurt the performance of the built trackers. To address this problem, we propose a multichannel Convolutional Neural Networks (CNN) architecture, in which we treat English and Chinese language as different input channels of one single CNN model. In the evaluation of DSTC5, we found that such multichannel architecture can effectively improve the robustness against translation errors. Additionally, our method for DSTC5 is purely machine learning based and requires no prior knowledge about the target language. We consider this a desirable property for building a tracker in the cross-language context, as not every developer will be familiar with both languages.

**URL:** https://arxiv.org/abs/1701.06247

**Notes:** CNN for dialog state tracking

### Learning to Decode for Future Success

**Authors:** Jiwei Li, Will Monroe, Dan Jurafsky

**Abstract:** We introduce a general strategy for improving neural sequence generation by incorporating knowledge about the future. Our decoder combines a standard sequence decoder with a `soothsayer' prediction function Q that estimates the outcome in the future of generating a word in the present. Our model draws on the same intuitions as reinforcement learning, but is both simpler and higher performing, avoiding known problems with the use of reinforcement learning in tasks with enormous search spaces like sequence generation. We demonstrate our model by incorporating Q functions that incrementally predict what the future BLEU or ROUGE score of the completed sequence will be, its future length, and the backwards probability of the source given the future target sequence. Experimental results show that future rediction yields improved performance in abstractive summarization and conversational response generation and the state-of-the-art in machine translation, while also enabling the decoder to generate outputs that have specific properties.

**URL:** https://arxiv.org/abs/1701.06549

**Notes:** future prediction with Q-learning

### Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

**Authors:** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean

**Abstract:** The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increasing model capacity without a proportional increase in computation. In practice, however, there are significant algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of conditional computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters. We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at lower computational cost.

**URL:** https://arxiv.org/abs/1701.06538

**Notes:** dropout analog from Dean & Hinton

### Regularizing Neural Networks by Penalizing Confident Output Distributions

**Authors:** Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, Geoffrey Hinton

**Abstract:** We systematically explore regularizing neural networks by penalizing low entropy output distributions. We show that penalizing low entropy output distributions, which has been shown to improve exploration in reinforcement learning, acts as a strong regularizer in supervised learning. Furthermore, we connect a maximum entropy based confidence penalty to label smoothing through the direction of the KL divergence. We exhaustively evaluate the proposed confidence penalty and label smoothing on 6 common benchmarks: image classification (MNIST and Cifar-10), language modeling (Penn Treebank), machine translation (WMT'14 English-to-German), and speech recognition (TIMIT and WSJ). We find that both label smoothing and the confidence penalty improve state-of-the-art models across benchmarks without modifying existing hyperparameters, suggesting the wide applicability of these regularizers.

**URL:** https://arxiv.org/abs/1701.06548

**Notes:** smart regularization from Hinton

### Discriminative Neural Topic Models

**Authors:** Gaurav Pandey, Ambedkar Dukkipati

**Abstract:** We propose a neural network based approach for learning topics from text and image datasets. The model makes no assumptions about the conditional distribution of the observed features given the latent topics. This allows us to perform topic modelling efficiently using sentences of documents and patches of images as observed features, rather than limiting ourselves to words. Moreover, the proposed approach is online, and hence can be used for streaming data. Furthermore, since the approach utilizes neural networks, it can be implemented on GPU with ease, and hence it is very scalable.

**URL:** https://arxiv.org/abs/1701.06796

**Notes:** don't like topic modeling but you should stay in touch with advances these days

