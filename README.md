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
   - Download, extract, and install Mosesdecoder at [this location](packages/mosesdecoder).
   - Download, extract, and install the Indic NLP library at [this location](packages/indic-nlp/indic_nlp_library). Also, extract [Indic NLP Resources](https://github.com/anoopkunchukuttan/indic_nlp_resources) at [this location](packages/indic-nlp/indic_nlp_resources). You can skip this step if you are not working with the Indic languages.  
   - Download, extract, and install the HuggingFace Transformers library at [this location](packages/transformers).
   ### 2.3 Installing Fairseq
   - Clone this repository. Its parent directory will act as the home directory for all the preprocessing, training, and evaluation scripts in this work.  
   - Run the following commands from the home directory. 
      ```
      cd 'work/systems/baseline-NMT/fairseq'
      pip install --editable ./
      ```
   - It will install the Baseline-NMT System based on the Fairseq library along with its dependencies. Note that we used multiple versions of the Fairseq systems located at [this location](work/systems/). So, we always used the exact paths of the training and evaluation files to avoid the conflicts.
   ### 2.4 Downloading and Extracting XLM-R
   - Download all the files associated with the XLM-R from the [HuggingFace hub](https://huggingface.co/xlm-roberta-base). Use the option *'List all files in model'* to view and download the files namely *config.json, pytorch_model.bin, sentencepiece.bpe.model,* and *tokenizer.json*.  	
   - Put these files in [this directory](/work/bert/models/pre-trained/xlm-roberta/xlmr.base).
## 3. Preprocessing<a name="3"></a>
   - Please visit the [Fairseq](https://fairseq.readthedocs.io/en/latest/) and [bert-nmt](https://github.com/bert-nmt/bert-nmt) libraries to get familiar with the basic preprocessing, training, and evaluation steps, as our work is built over them.
   - We used Mosesdecoder to preprocess the English datasets, but switched to the Indic NLP library for the Indic languages such as Hindi, Gujarati, Bengali, and Marathi.  
   - We used the Sentencepiece BPE for the word segmentation. When the source and target languages shared substantial characters, we processed the datasets with the joint BPE using [this script](work/scripts/preprocessing/tokenize-bpe-shared.sh). Otherwise, a different [script](work/scripts/preprocessing/tokenize-bpe-seperate.sh) was used.
   - For the English-Hindi dataset used to demonstrate this work, we used the latter script.
   - Set the *HOME_DIR* to the parent directory of this repository.
   - We have already put train, test, and dev files at the *RAW_DATA_DIR*. You can change them with your files with the same naming conventions.
   - We merged the training data with the massive monolingual datasets to learn better BPE segmentation. Put these datasets at *RAW_MONOLINGUAL_DATA_DIR*. We used massive [OSCAR corpus](https://oscar-corpus.com/) in our work, but for this demo, we just used the same train files.
   - Switch between the Indic NLP or Moses library based on the languages by commenting out the *clean_norm_tok* function, as shown in the script.   
   - Run this script which preprocesses all the files and saves at *PREPROCESSED_DATA_DIR*. *tokenized-BPE* directory contains all the intermediate files after normalization, tokenization, etc., as well as, all the final BPEd files.
   - Then, this script binarises the data to be used by Fairseq based systems, and saves in the *binary* directory. It uses the Fairseq binariser from the [xlm-r-fused system](/work/systems/xlm-r-fused/bert-nmt/) to binarise the files for the baseline, as well as, the XLM-R-fused systems. It uses *--bert-model-name* to access the XLM-R tokenizer to tokenize the source files, as they were also used by the XLM-R component along with the standard NMT-encoder in the XLM-R-fused systems.
   - (Optional) Note that this system is primarily based upon the XLM-R, but we can use other masked language models provided by the Huggingface Transformers library. We need to make some changes as follows:
      - Download and extract the new language model as mentioned in Step 2.4 
      - Import the corresponding Tokenizer and Model from the HuggingFace Transformers library in the XLM-R-fused system with the default one as mentioned below:      
         ```
         from transformers import XLMRobertaModel
         BertModel = XLMRobertaModel

         from transformers import XLMRobertaTokenizer
         BertTokenizer = XLMRobertaTokenizer
         ``` 
      - We need to import them in the following files:
	     - work/systems/xlm-r-fused/bert-nmt/preprocess.py
	     - work/systems/xlm-r-fused/bert-nmt/interactive.py
	     - work/systems/xlm-r-fused/bert-nmt/fairseq_cli/preprocess.py
	     - work/systems/xlm-r-fused/bert-nmt/fairseq_cli/interactive.py
	     - work/systems/xlm-r-fused/bert-nmt/fairseq/tasks/translation.py
	     - work/systems/xlm-r-fused/bert-nmt/fairseq/models/transformer.py
	     - work/systems/xlm-r-fused/bert-nmt/fairseq/binarizer.py
      - Further, we need to change the start `(<s>)` and end `(</s>)` tokens in these files as per the new language model.

## <a name="4"></a>4. Baseline NMT System
   ### 4.1 Training baseline NMT system
   - Train the baseline system with [this script](work/scripts/baseline/train_baseline.sh).
   - It will accumulate the gradients to form a larger effective batch size. Batch size = (number of GPUs) * (*--max-tokens*) * (*--update-freq*).
   - It uses an early stopping validation strategy with *--patience* determining the maximum number of checkpoints with declining BLEU scores.
   - Our work uses the Transformer architecture: *transformer_iwslt_de_en* as default. The XLM-R-fused systems restore the parameters from the baseline systems, so their architectures should match. You can also use other larger architectures, but you need to give the same underlying architecture for the XLM-R-fused systems as well. Check [this file](work/systems/xlm-r-fused/bert-nmt/fairseq/models/transformer.py) for additional architectures implementing the attention based fusion.
   - It saves the checkpoints at *BASELINE_NMT_CHECKPOINTS_DIR*. 
   ### 4.2 Evaluating baseline NMT system
   - Evaluate the baseline system with [this script](work/scripts/baseline/eval_inter_baseline.sh)
   - We need to evaluate BPEd test file with the best checkpoint. Use *--remove-bpe=sentencepiece* to remove the BPE segmentation from the output file. 
   - Use either Indic NLP or Moses to detokenize the output file as shown in the script.
   - This script calculates the final BLEU scores using SacreBLEU using the untouched test file of the target language.
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
