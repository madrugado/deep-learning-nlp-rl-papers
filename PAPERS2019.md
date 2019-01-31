
Table of Contents
=================

* [Articles](#articles)
  * [2019\-01](#2019-01)
    * [Pull out all the stops: Textual analysis via punctuation sequences](#pull-out-all-the-stops-textual-analysis-via-punctuation-sequences)
    * [Assessing BERT’s Syntactic Abilities](#assessing-berts-syntactic-abilities)
    * [Human few\-shot learning of compositional instructions](#human-few-shot-learning-of-compositional-instructions)
    * [No Training Required: Exploring Random Encoders for Sentence Classification](#no-training-required-exploring-random-encoders-for-sentence-classification)
    * [Pay Less Attention with Lightweight and Dynamic Convolutions](#pay-less-attention-with-lightweight-and-dynamic-convolutions)

Articles
========
## 2019-01
### Pull out all the stops: Textual analysis via punctuation sequences

**Authors:** Alexandra N. M. Darmon, Marya Bazzi, Sam D. Howison, Mason A. Porter

**Abstract:** Whether enjoying the lucid prose of a favorite author or slogging through some other writer's cumbersome, heavy-set prattle (full of parentheses, em-dashes, compound adjectives, and Oxford commas), readers will notice stylistic signatures not only in word choice and grammar, but also in punctuation itself. Indeed, visual sequences of punctuation from different authors produce marvelously different (and visually striking) sequences. Punctuation is a largely overlooked stylistic feature in "stylometry'', the quantitative analysis of written text. In this paper, we examine punctuation sequences in a corpus of literary documents and ask the following questions: Are the properties of such sequences a distinctive feature of different authors? Is it possible to distinguish literary genres based on their punctuation sequences? Do the punctuation styles of authors evolve over time? Are we on to something interesting in trying to do stylometry without words, or are we full of sound and fury (signifying nothing)?

**URL:** https://arxiv.org/abs/1901.00519

**Notes:** really nice idea - analyze the punctuation itself, it is shown to be enough to distinct authorship; I think that some other tasks could be formulated, like punctuation style transfer or punctuation improvement only just from big corpora

### Assessing BERT’s Syntactic Abilities

**Authors:** Yoav Goldberg

**Abstract:** I assess the extent to which the recently introduced BERT model captures English syntactic phenomena, using (1) naturally-occurring subject-verb agreement stimuli; (2) “coloreless green ideas” subject-verb agreement stimuli, in which content words in natural sentences are randomly replaced with words sharing the same part-of-speech and inflection; and (3) manually crafted stimuli for subject-verb agreement and reflexive anaphora phenomena. The BERT model performs remarkably well on all cases.

**URL:** http://u.cs.biu.ac.il/~yogo/bert-syntax.pdf

**Notes:** I like the idea of this small and concise research, it answers clear question clearly; I think more research could be done in this direction

### Human few-shot learning of compositional instructions

**Authors:** Brenden M. Lake, Tal Linzen, Marco Baroni

**Abstract:** People learn in fast and flexible ways that have not been emulated by machines. Once a person learns a new verb "dax," he or she can effortlessly understand how to "dax twice," "walk and dax," or "dax vigorously." There have been striking recent improvements in machine learning for natural language processing, yet the best algorithms require vast amounts of experience and struggle to generalize new concepts in compositional ways. To better understand these distinctively human abilities, we study the compositional skills of people through language-like instruction learning tasks. Our results show that people can learn and use novel functional concepts from very few examples (few-shot learning), successfully applying familiar functions to novel inputs. People can also compose concepts in complex ways that go beyond the provided demonstrations. Two additional experiments examined the assumptions and inductive biases that people make when solving these tasks, revealing three biases: mutual exclusivity, one-to-one mappings, and iconic concatenation. We discuss the implications for cognitive modeling and the potential for building machines with more human-like language learning capabilities.

**URL:** https://arxiv.org/abs/1901.04587

**Notes:** interesting work on few shot learning in language; a person should "translate" from unknown constructed language to visual language; some flaws: there are only United State residents (so English-speaking) and proposed tasks could influence each other

### No Training Required: Exploring Random Encoders for Sentence Classification

**Authors:** John Wieting, Douwe Kiela

**Abstract:** We explore various methods for computing sentence representations from pre-trained word embeddings without any training, i.e., using nothing but random parameterizations. Our aim is to put sentence embeddings on more solid footing by 1) looking at how much modern sentence embeddings gain over random methods---as it turns out, surprisingly little; and by 2) providing the field with more appropriate baselines going forward---which are, as it turns out, quite strong. We also make important observations about proper experimental protocol for sentence classification evaluation, together with recommendations for future research.

**URL:** https://arxiv.org/abs/1901.10444

**Notes:** new work from FAIR about random encoders for text clf; pooling over random projection of word emb, randomly init'ed (and never updated) LSTMs, and analog of simple RNN, also random; LSTM even reach a SotA on TREC, and they all are really good in all tasks

### Pay Less Attention with Lightweight and Dynamic Convolutions

**Authors:** Felix Wu, Angela Fan, Alexei Baevski, Yann N. Dauphin, Michael Auli

**Abstract:** Self-attention is a useful mechanism to build generative models for language and images. It determines the importance of context elements by comparing each element to the current time step. In this paper, we show that a very lightweight convolution can perform competitively to the best reported self-attention results. Next, we introduce dynamic convolutions which are simpler and more efficient than self-attention. We predict separate convolution kernels based solely on the current time-step in order to determine the importance of context elements. The number of operations required by this approach scales linearly in the input length, whereas self-attention is quadratic. Experiments on large-scale machine translation, language modeling and abstractive summarization show that dynamic convolutions improve over strong self-attention models. On the WMT'14 English-German test set dynamic convolutions achieve a new state of the art of 29.7 BLEU.

**URL:** https://arxiv.org/abs/1901.10430

**Notes:** Facebook takes a next step in quasi-RNNs: lightweight convs are using softmax pooling over time, and dynamic convs use position encoding to shift weights for particular timestep; this work achieves new SotA on En-De MT and also they're close in other tasks

