
Table of Contents
=================

  * [Miscellaneous](#miscellaneous)
    * [Lecture notes](#lecture-notes)
      * [Monte Carlo Methods and Importance Sampling](#monte-carlo-methods-and-importance-sampling)
      * [Kernel Canonical Correlation Analysis](#kernel-canonical-correlation-analysis)
    * [Blueprints](#blueprints)
      * [In\-Datacenter Performance Analysis of a Tensor Processing Unit​](#in-datacenter-performance-analysis-of-a-tensor-processing-unit)
      * [TensorFlow: Large\-Scale Machine Learning on Heterogeneous Distributed Systems](#tensorflow-large-scale-machine-learning-on-heterogeneous-distributed-systems)
      * [DyNet: The Dynamic Neural Network Toolkit](#dynet-the-dynamic-neural-network-toolkit)
      * [AllenNLP: A Deep Semantic Natural Language Processing Platform](#allennlp-a-deep-semantic-natural-language-processing-platform)
    * [Reports/Surveys](#reportssurveys)
      * [Best Practices for Applying Deep Learning to Novel Applications](#best-practices-for-applying-deep-learning-to-novel-applications)
      * [Automatic Keyword Extraction for Text Summarization: A Survey](#automatic-keyword-extraction-for-text-summarization-a-survey)
      * [Factorization tricks for LSTM networks](#factorization-tricks-for-lstm-networks)
      * [Symbolic, Distributed and Distributional Representations for Natural Language Processing in the Era of Deep Learning: a Survey](#symbolic-distributed-and-distributional-representations-for-natural-language-processing-in-the-era-of-deep-learning-a-survey)
      * [Deep Reinforcement Learning: An Overview](#deep-reinforcement-learning-an-overview)
      * [Algorithms for multi\-armed bandit problems](#algorithms-for-multi-armed-bandit-problems)
      * [A comparison of Extrinsic Clustering Evaluation Metrics based on Formal Constraints](#a-comparison-of-extrinsic-clustering-evaluation-metrics-based-on-formal-constraints)
    * [Other](#other)
      * [Living Together: Mind and Machine Intelligence](#living-together-mind-and-machine-intelligence)
      * [Machine Teaching: A New Paradigm for Building Machine Learning Systems](#machine-teaching-a-new-paradigm-for-building-machine-learning-systems)
    * [Tutorial](#tutorial)
      * [NIPS 2016 Tutorial: Generative Adversarial Networks](#nips-2016-tutorial-generative-adversarial-networks)

Miscellaneous
=============
## Lecture notes
### Monte Carlo Methods and Importance Sampling

**Authors:** Eric C. Anderson, E. A. Thompson

**Abstract:** Lecture Notes for Stat 578c Statistical Genetics

**URL:** http://ib.berkeley.edu/labs/slatkin/eriq/classes/guest_lect/mc_lecture_notes.pdf

**Notes:** simple explanation for importance sampling in stats, IS in softmax is coming from here

### Kernel Canonical Correlation Analysis

**Authors:** Max Welling

**Abstract:** Kernel Canonical Correlation Analysis

**URL:** http://www.ics.uci.edu/~welling/classnotes/papers_class/kCCA.pdf

**Notes:** explanation of kernel Canonical Correlation Analysis from Max Welling

## Blueprints
### In-Datacenter Performance Analysis of a Tensor Processing Unit​

**Authors:** Norman P. Jouppi et al.

**Abstract:** Many architects believe that major improvements in cost-energy-performance must now come from domain-specific hardware. This paper evaluates a custom ASIC—called a ​Tensor Pro​cessing Unit (TPU)— deployed in datacenters since 2015 that accelerates the inference phase of neural networks (NN). The heart of the TPU is a 65,536 8-bit MAC matrix multiply unit that offers a peak throughput of 92 TeraOps/second (TOPS) and a large (28 MiB) software-managed on-chip memory. The TPU’s deterministic execution model is a better match to the 99th-percentile response-time requirement of our NN applications than are the time-varying optimizations of CPUs and GPUs (caches, out-of-order execution, multithreading, multiprocessing, prefetching, ...) that help average throughput more than guaranteed latency. The lack of such features helps explain why, despite having myriad MACs and a big memory, the TPU is relatively small and low power. We compare the TPU to a server-class Intel Haswell CPU and an Nvidia K80 GPU, which are contemporaries deployed in the same datacenters. Our workload, written in the high-level TensorFlow framework, uses production NN applications (MLPs, CNNs, and LSTMs) that represent 95% of our datacenters’ NN inference demand. Despite low utilization for some applications, the TPU is on average about 15X - 30X faster than its contemporary GPU or CPU, with TOPS/Watt about 30X - 80X higher. Moreover, using the GPU’s GDDR5 memory in the TPU would triple achieved TOPS and raise TOPS/Watt to nearly 70X the GPU and 200X the CPU.

**URL:** https://drive.google.com/file/d/0Bx4hafXDDq2EMzRNcy1vSUxtcEk/view

**Notes:** a blueprint about new Google TPUs; fascinating future of Deep Learning

### TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems

**Authors:** Martín Abadi et al.

**Abstract:** TensorFlow is an interface for expressing machine learning algorithms, and an implementation for executing such algorithms. A computation expressed using TensorFlow can be executed with little or no change on a wide variety of heterogeneous systems, ranging from mobile devices such as phones and tablets up to large-scale distributed systems of hundreds of machines and thousands of computational devices such as GPU cards. The system is flexible and can be used to express a wide variety of algorithms, including training and inference algorithms for deep neural network models, and it has been used for conducting research and for deploying machine learning systems into production across more than a dozen areas of computer science and other fields, including speech recognition, computer vision, robotics, information retrieval, natural language processing, geographic information extraction, and computational drug discovery. This paper describes the TensorFlow interface and an implementation of that interface that we have built at Google. The TensorFlow API and a reference implementation were released as an open-source package under the Apache 2.0 license in November, 2015 and are available at www.tensorflow.org.

**URL:** http://download.tensorflow.org/paper/whitepaper2015.pdf

**Notes:** long time missed here blueprint on Tensorflow

### DyNet: The Dynamic Neural Network Toolkit

**Authors:** Graham Neubig, Chris Dyer, Yoav Goldberg, Austin Matthews, Waleed Ammar, Antonios Anastasopoulos, Miguel Ballesteros, David Chiang, Daniel Clothiaux, Trevor Cohn, Kevin Duh, Manaal Faruqui, Cynthia Gan, Dan Garrette, Yangfeng Ji, Lingpeng Kong, Adhiguna Kuncoro, Gaurav Kumar, Chaitanya Malaviya, Paul Michel, Yusuke Oda, Matthew Richardson, Naomi Saphra, Swabha Swayamdipta, Pengcheng Yin

**Abstract:** We describe DyNet, a toolkit for implementing neural network models based on dynamic declaration of network structure. In the static declaration strategy that is used in toolkits like Theano, CNTK, and TensorFlow, the user first defines a computation graph (a symbolic representation of the computation), and then examples are fed into an engine that executes this computation and computes its derivatives. In DyNet's dynamic declaration strategy, computation graph construction is mostly transparent, being implicitly constructed by executing procedural code that computes the network outputs, and the user is free to use different network structures for each input. Dynamic declaration thus facilitates the implementation of more complicated network architectures, and DyNet is specifically designed to allow users to implement their models in a way that is idiomatic in their preferred programming language (C++ or Python). One challenge with dynamic declaration is that because the symbolic computation graph is defined anew for every training example, its construction must have low overhead. To achieve this, DyNet has an optimized C++ backend and lightweight graph representation. Experiments show that DyNet's speeds are faster than or comparable with static declaration toolkits, and significantly faster than Chainer, another dynamic declaration toolkit. DyNet is released open-source under the Apache 2.0 license and available at this http URL

**URL:** https://arxiv.org/abs/1701.03980

**Notes:** The paper has remarkable list of authors - DeepMind, Google, IBM Watson, CMU, AI2 & MSR... New DNN framework.

### AllenNLP: A Deep Semantic Natural Language Processing Platform

**Authors:** Matt Gardner, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson Liu, Matthew Peters, Michael Schmitz, Luke Zettlemoyer

**Abstract:** This paper describes AllenNLP, a platform for research on deep learning methods in natural language understanding. AllenNLP is designed to support researchers who want to build novel language understanding models quickly and easily. It is built on top of PyTorch, allowing for dynamic computation graphs, and provides (1) a flexible data API that handles intelligent batching and padding, (2) highlevel abstractions for common operations in working with text, and (3) a modular and extensible experiment framework that makes doing good science easy. It also includes reference implementations of high quality approaches for both core semantic problems (e.g. semantic role labeling (Palmer et al., 2005)) and language understanding applications (e.g. machine comprehension (Rajpurkar et al., 2016)). AllenNLP is an ongoing open-source effort maintained by engineers and researchers at the Allen Institute for Artificial Intelligence.

**URL:** http://allennlp.org/papers/AllenNLP_white_paper.pdf

**Notes:** white paper for freshly presented AllenNLP - DL platform for NLP tasks; made on PyTorch

## Reports/Surveys
### Best Practices for Applying Deep Learning to Novel Applications

**Authors:** Leslie N. Smith

**Abstract:** This report is targeted to groups who are subject matter experts in their application but deep learning novices. It contains practical advice for those interested in testing the use of deep neural networks on applications that are novel for deep learning. We suggest making your project more manageable by dividing it into phases. For each phase this report contains numerous recommendations and insights to assist novice practitioners.

**URL:** https://arxiv.org/abs/1704.01568

**Notes:** some notes on applying DL to new areas

### Automatic Keyword Extraction for Text Summarization: A Survey

**Authors:** Santosh Kumar Bharti, Korra Sathya Babu

**Abstract:** In recent times, data is growing rapidly in every domain such as news, social media, banking, education, etc. Due to the excessiveness of data, there is a need of automatic summarizer which will be capable to summarize the data especially textual data in original document without losing any critical purposes. Text summarization is emerged as an important research area in recent past. In this regard, review of existing work on text summarization process is useful for carrying out further research. In this paper, recent literature on automatic keyword extraction and text summarization are presented since text summarization process is highly depend on keyword extraction. This literature includes the discussion about different methodology used for keyword extraction and text summarization. It also discusses about different databases used for text summarization in several domains along with evaluation matrices. Finally, it discusses briefly about issues and research challenges faced by researchers along with future direction.

**URL:** https://arxiv.org/abs/1704.03242

**Notes:** useful list of works in keyword extraction

### Factorization tricks for LSTM networks

**Authors:** Oleksii Kuchaiev, Boris Ginsburg

**Abstract:** We present two simple ways of reducing the number of parameters and accelerating the training of large Long Short-Term Memory (LSTM) networks: the first one is "matrix factorization by design" of LSTM matrix into the product of two smaller matrices, and the second one is partitioning of LSTM matrix, its inputs and states into the independent groups. Both approaches allow us to train large LSTM networks significantly faster to the state-of the art perplexity. On the One Billion Word Benchmark we improve single model perplexity down to 23.36.

**URL:** https://arxiv.org/abs/1703.10722

**Notes:** could be useful bunch of tricks for LSTM from NVIDIA engineers

### Symbolic, Distributed and Distributional Representations for Natural Language Processing in the Era of Deep Learning: a Survey

**Authors:** Lorenzo Ferrone, Fabio Massimo Zanzotto

**Abstract:** Natural language and symbols are intimately correlated. Recent advances in machine learning (ML) and in natural language processing (NLP) seem to contradict the above intuition: symbols are fading away, erased by vectors or tensors called distributed and distributional representations. However, there is a strict link between distributed/distributional representations and symbols, being the first an approximation of the second. A clearer understanding of the strict link between distributed/distributional representations and symbols will certainly lead to radically new deep learning networks. In this paper we make a survey that aims to draw the link between symbolic representations and distributed/distributional representations. This is the right time to revitalize the area of interpreting how symbols are represented inside neural networks.

**URL:** https://arxiv.org/abs/1702.00764

**Notes:** review of nlp representations

### Deep Reinforcement Learning: An Overview

**Authors:** Yuxi Li

**Abstract:** We give an overview of recent exciting achievements of deep reinforcement learning (RL). We start with background of deep learning and reinforcement learning, as well as introduction of testbeds. Next we discuss Deep Q-Network (DQN) and its extensions, asynchronous methods, policy optimization, reward, and planning. After that, we talk about attention and memory, unsupervised learning, and learning to learn. Then we discuss various applications of RL, including games, in particular, AlphaGo, robotics, spoken dialogue systems (a.k.a. chatbot), machine translation, text sequence prediction, neural architecture design, personalized web services, healthcare, finance, and music generation. We mention topics/papers not reviewed yet. After listing a collection of RL resources, we close with discussions.

**URL:** https://arxiv.org/abs/1701.07274

**Notes:** RL overview, including dialog systems

### Algorithms for multi-armed bandit problems

**Authors:** Volodymyr Kuleshov, Doina Precup

**Abstract:** Although many algorithms for the multi-armed bandit problem are well-understood theoretically, empirical confirmation of their effectiveness is generally scarce. This paper presents a thorough empirical study of the most popular multi-armed bandit algorithms. Three important observations can be made from our results. Firstly, simple heuristics such as epsilon-greedy and Boltzmann exploration outperform theoretically sound algorithms on most settings by a significant margin. Secondly, the performance of most algorithms varies dramatically with the parameters of the bandit problem. Our study identifies for each algorithm the settings where it performs well, and the settings where it performs poorly. Thirdly, the algorithms' performance relative each to other is affected only by the number of bandit arms and the variance of the rewards. This finding may guide the design of subsequent empirical evaluations. In the second part of the paper, we turn our attention to an important area of application of bandit algorithms: clinical trials. Although the design of clinical trials has been one of the principal practical problems motivating research on multi-armed bandits, bandit algorithms have never been evaluated as potential treatment allocation strategies. Using data from a real study, we simulate the outcome that a 2001-2002 clinical trial would have had if bandit algorithms had been used to allocate patients to treatments. We find that an adaptive trial would have successfully treated at least 50% more patients, while significantly reducing the number of adverse effects and increasing patient retention. At the end of the trial, the best treatment could have still been identified with a high level of statistical confidence. Our findings demonstrate that bandit algorithms are attractive alternatives to current adaptive treatment allocation strategies.

**URL:** https://arxiv.org/abs/1402.6028

**Notes:** an pretty old (2014) but seems to useful review of k-armed bandits algos

### A comparison of Extrinsic Clustering Evaluation Metrics based on Formal Constraints

**Authors:** Enrique Amigo, Julio Gonzalo, Javier Artiles, Felisa Verdejo

**Abstract:** There is a wide set of evaluation metrics available to compare the quality of text clustering algorithms. In this article, we define a few intuitive formal constraints on such metrics which shed light on which aspects of the quality of a clustering are captured by different metric families. These formal constraints are validated in an experiment involving human assessments, and compared with other constraints proposed in the literature. Our analysis of a wide range of metrics shows that only BCubed satisfies all formal constraints. We also extend the analysis to the problem of overlapping clustering, where items can simultaneously belong to more than one cluster. As BCubed cannot be directly applied to this task, we propose a modifiedversion of Bcubed that avoids the problems found with other metrics.

**URL:** http://nlp.uned.es/docs/amigo2007a.pdf

**Notes:** comparison of clustering metrics for texts

## Other
### Living Together: Mind and Machine Intelligence

**Authors:** Neil D. Lawrence

**Abstract:** In this paper we consider the nature of the machine intelligences we have created in the context of our human intelligence. We suggest that the fundamental difference between human and machine intelligence comes down to \emph{embodiment factors}. We define embodiment factors as the ratio between an entity's ability to communicate information vs compute information. We speculate on the role of embodiment factors in driving our own intelligence and consciousness. We briefly review dual process models of cognition and cast machine intelligence within that framework, characterising it as a dominant System Zero, which can drive behaviour through interfacing with us subconsciously. Driven by concerns about the consequence of such a system we suggest prophylactic courses of action that could be considered. Our main conclusion is that it is \emph{not} sentient intelligence we should fear but \emph{non-sentient} intelligence.

**URL:** https://arxiv.org/abs/1705.07996

**Notes:** Jack Clark recommends this philosophical paper; human brain still better than computer in compression of input data by a several orders of magnitude; it is nice since we (the computer scientists to whom I have the courage to attribute myself) have a lot of stuff to do before the singularity

### Machine Teaching: A New Paradigm for Building Machine Learning Systems

**Authors:** Patrice Y. Simard, Saleema Amershi, David M. Chickering, Alicia Edelman Pelton, Soroush Ghorashi, Christopher Meek, Gonzalo Ramos, Jina Suh, Johan Verwey, Mo Wang, John Wernsing

**Abstract:** The current processes for building machine learning systems require practitioners with deep knowledge of machine learning. This significantly limits the number of machine learning systems that can be created and has led to a mismatch between the demand for machine learning systems and the ability for organizations to build them. We believe that in order to meet this growing demand for machine learning systems we must significantly increase the number of individuals that can teach machines. We postulate that we can achieve this goal by making the process of teaching machines easy, fast and above all, universally accessible. While machine learning focuses on creating new algorithms and improving the accuracy of "learners", the machine teaching discipline focuses on the efficacy of the "teachers". Machine teaching as a discipline is a paradigm shift that follows and extends principles of software engineering and programming languages. We put a strong emphasis on the teacher and the teacher's interaction with data, as well as crucial components such as techniques and design principles of interaction and visualization. In this paper, we present our position regarding the discipline of machine teaching and articulate fundamental machine teaching principles. We also describe how, by decoupling knowledge about machine learning algorithms from the process of teaching, we can accelerate innovation and empower millions of new uses for machine learning models.

**URL:** https://arxiv.org/abs/1707.06742

**Notes:** Microsoft proposes a way to handle Machine Learning as Software Development, for me it is not that obvious why do we need to state a new field of study, apart of Software Development, but the stated problem does exist in the my day-to-day live too

## Tutorial
### NIPS 2016 Tutorial: Generative Adversarial Networks

**Authors:** Ian Goodfellow

**Abstract:** This report summarizes the tutorial presented by the author at NIPS 2016 on generative adversarial networks (GANs). The tutorial describes: (1) Why generative modeling is a topic worth studying, (2) how generative models work, and how GANs compare to other generative models, (3) the details of how GANs work, (4) research frontiers in GANs, and (5) state-of-the-art image models that combine GANs with other methods. Finally, the tutorial contains three exercises for readers to complete, and the solutions to these exercises.

**URL:** https://arxiv.org/abs/1701.00160

**Notes:** Goodfellow's tutorial couldn't hurt

