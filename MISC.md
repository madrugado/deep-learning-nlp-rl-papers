
Table of Contents
=================

  * [Miscellaneous](#miscellaneous)
    * [Lecture notes](#lecture-notes)
      * [Monte Carlo Methods and Importance Sampling](#monte-carlo-methods-and-importance-sampling)
    * [Blueprints](#blueprints)
      * [In\-Datacenter Performance Analysis of a Tensor Processing Unit​](#in-datacenter-performance-analysis-of-a-tensor-processing-unit)
    * [Reports](#reports)
      * [Best Practices for Applying Deep Learning to Novel Applications](#best-practices-for-applying-deep-learning-to-novel-applications)

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

## Reports
### Best Practices for Applying Deep Learning to Novel Applications

**Authors:** Leslie N. Smith

**Abstract:** This report is targeted to groups who are subject matter experts in their application but deep learning novices. It contains practical advice for those interested in testing the use of deep neural networks on applications that are novel for deep learning. We suggest making your project more manageable by dividing it into phases. For each phase this report contains numerous recommendations and insights to assist novice practitioners.

**URL:** https://arxiv.org/abs/1704.01568

**Notes:** some notes on applying DL to new areas

