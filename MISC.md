
Table of Contents
=================

  * [Miscellaneous](#miscellaneous)
    * [Lecture notes](#lecture-notes)
      * [Monte Carlo Methods and Importance Sampling](#monte-carlo-methods-and-importance-sampling)
    * [Blueprints](#blueprints)
      * [In\-Datacenter Performance Analysis of a Tensor Processing Unit​](#in-datacenter-performance-analysis-of-a-tensor-processing-unit)
      * [TensorFlow: Large\-Scale Machine Learning on Heterogeneous Distributed Systems](#tensorflow-large-scale-machine-learning-on-heterogeneous-distributed-systems)
    * [Reports/Surveys](#reportssurveys)
      * [Best Practices for Applying Deep Learning to Novel Applications](#best-practices-for-applying-deep-learning-to-novel-applications)
      * [Automatic Keyword Extraction for Text Summarization: A Survey](#automatic-keyword-extraction-for-text-summarization-a-survey)
      * [Factorization tricks for LSTM networks](#factorization-tricks-for-lstm-networks)
    * [Other](#other)
      * [Living Together: Mind and Machine Intelligence](#living-together-mind-and-machine-intelligence)

Miscellaneous
=============
## Lecture notes
### Monte Carlo Methods and Importance Sampling

**Authors:** Eric C. Anderson, E. A. Thompson

**Abstract:** Lecture Notes for Stat 578c Statistical Genetics

**URL:** http://ib.berkeley.edu/labs/slatkin/eriq/classes/guest_lect/mc_lecture_notes.pdf

**Notes:** simple explanation for importance sampling in stats, IS in softmax is coming from here

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

## Other
### Living Together: Mind and Machine Intelligence

**Authors:** Neil D. Lawrence

**Abstract:** In this paper we consider the nature of the machine intelligences we have created in the context of our human intelligence. We suggest that the fundamental difference between human and machine intelligence comes down to \emph{embodiment factors}. We define embodiment factors as the ratio between an entity's ability to communicate information vs compute information. We speculate on the role of embodiment factors in driving our own intelligence and consciousness. We briefly review dual process models of cognition and cast machine intelligence within that framework, characterising it as a dominant System Zero, which can drive behaviour through interfacing with us subconsciously. Driven by concerns about the consequence of such a system we suggest prophylactic courses of action that could be considered. Our main conclusion is that it is \emph{not} sentient intelligence we should fear but \emph{non-sentient} intelligence.

**URL:** https://arxiv.org/abs/1705.07996

**Notes:** Jack Clark recommends this philosophical paper; human brain still better than computer in compression of input data by a several orders of magnitude; it is nice since we (the computer scientists to whom I have the courage to attribute myself) have a lot of stuff to do before the singularity

