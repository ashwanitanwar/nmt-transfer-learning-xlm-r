***To be updated***

This repository presents the work done during my master's thesis with the title **"Improving Low-Resource Neural Machine Translation of Related Languages by Transfer Learning"** advised by [Alexandra Birch](https://homepages.inf.ed.ac.uk/abmayne/) and [Rachel Bawden](https://rbawden.github.io/) at The University of Edinburgh. It was submitted in August 2020. It investigated some of the Transfer Learning approaches for the Neural Machine Translation (NMT) systems by utilising the masked language models such as XLM-RoBERTa [[1]](#ref1). This project was primarily built over the attention based fusion of the contextualisd word representations from the masked language models (MLM) with the Transformer based NMT system [[2]](#ref2). It also empirically traced the transfer of the syntactic knowledge by an analysis of the attention heads in this system based on the work of Reference [[3]](#ref3). This guide gives the necessary installation instructions along with a small working example using a small subset of the English-Hindi parallel dataset [[4]](#ref4). All the hyperparameters used in this work can be accessed from the thesis.

## Table of Contents
1. [An Overview of Attention-based Fusion](#1) 
2. [Installation](#2)
3. [Preprocessing](#3)
4. [Baseline NMT System](#4)
5. [XLM-R-fused NMT System](#5)
6. [Finetuning XLM-R](#6)
7. [Script Conversion](#7)
8. [Syntactic Analysis](#8)
9. [Additional Info](#9)
10. [References](#10)

## <a name="1"></a>1. An Overview of Attention-based Fusion
## <a name="2"></a>2. Installation
## <a name="3"></a>3. Preprocessing
## <a name="4"></a>4. Baseline NMT System
## <a name="5"></a>5. XLM-R-fused NMT System
## <a name="6"></a>6. Finetuning XLM-R
## <a name="7"></a>7. Script Conversion
## <a name="8"></a>8. Syntactic Analysis
## <a name="9"></a>9. Additional Info
## <a name="10"></a>10. References
<a id="ref1">[1]</a> [Conneau, Alexis, et al. "Unsupervised Cross-Lingual Representation Learning At Scale." arXiv preprint arXiv:1911.02116 (2019)](https://arxiv.org/pdf/1911.02116.pdf)

<a id="ref2">[2]</a> [Zhu, Jinhua, et al. "Incorporating BERT into Neural Machine Translation." International Conference on Learning Representations. 2019](https://openreview.net/pdf?id=Hyl7ygStwB)

<a id="ref3">[3]</a> [Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019, August). What Does BERT Look at? An Analysis of BERT’s Attention. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP (pp. 276-286).](https://www.aclweb.org/anthology/W19-4828.pdf)

<a id="ref4">[4]</a> [Anoop Kunchukuttan, Pratik Mehta, Pushpak Bhattacharyya. The IIT Bombay English-Hindi Parallel Corpus. Language Resources and Evaluation Conference. 2018. ](http://www.cfilt.iitb.ac.in/iitb_parallel/lrec2018_iitbparallel.pdf)
