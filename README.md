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

## 1. An Overview of Attention-based Fusion<a name="1"></a>
## 2. Installation<a name="2"></a>
   ### 2.1. Requirements
   - Python >= 3.5
   - PyTorch >= 1.5.0
   - [HuggingFace Transformers](https://github.com/huggingface/transformers) == 2.11.0
   - Tensorflow == 1.13.1
   - [Sacrebleu](https://github.com/mjpost/sacrebleu) >= 1.4.10
   - [Sentencepiece](https://github.com/google/sentencepiece) >= 0.1.91
   - [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) >= 0.6
   - [Mosesdecoder](https://github.com/moses-smt/mosesdecoder)
   ### 2.2  Installation in the packages directory
   - Download, extract, and install Mosesdecoder at this [location](packages/mosesdecoder).
   - Download, extract, and install the Indic NLP library at this [location](packages/indic-nlp/indic_nlp_library). Also, extract [Indic NLP Resources](https://github.com/anoopkunchukuttan/indic_nlp_resources) at this [location](packages/indic-nlp/indic_nlp_resources). You can skip this step if you are not working with the Indic languages.  
   - Download, extract, and install the HuggingFace Transformers library at this [location](packages/transformers).
   ### 2.3 Installing Fairseq
   - Clone this repository. Its parent directory will act as the home directory for all the preprocessing, training, and evaluation scripts in this work.  
   - Run the following commands from the home directory. 
   ```
   cd 'work/systems/baseline-NMT/fairseq'
   pip install --editable ./
   ```
   - It will install the Baseline-NMT System based on the Fairseq library along with its dependencies. Note that we used multiple versions of the Fairseq systems located at this [location](work/systems/). So, we always used the exact paths of the training and evaluation files to avoid the conflicts.
   ### 2.4 Downloading and Extracting XLM-R
   - Download all the files associated with the XLM-R from the [HuggingFace hub](https://huggingface.co/xlm-roberta-base). Use the option *'List all files in model'* to view and download the files namely *config.json, pytorch_model.bin, sentencepiece.bpe.model,* and *tokenizer.json*.  	
   - Put these files in [this directory](/work/bert/models/pre-trained/xlm-roberta/xlmr.base).
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
