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
   - Train the Transformer based baseline system with [this script](work/scripts/baseline/train_baseline.sh).
   - It will accumulate the gradients to form a larger effective batch size. Batch size = (number of GPUs) * (*--max-tokens*) * (*--update-freq*).
   - It uses an early stopping validation strategy with *--patience* determining the maximum number of checkpoints with declining BLEU scores.
   - Our work uses the Transformer architecture: *transformer_iwslt_de_en* as default. The XLM-R-fused systems restore the parameters from the baseline systems, so their architectures should match. You can also use other larger architectures, but you need to give the same underlying architecture for the XLM-R-fused systems as well. Check [this file](work/systems/xlm-r-fused/bert-nmt/fairseq/models/transformer.py) for additional architectures implementing the attention based fusion.
   - It saves the checkpoints at *BASELINE_NMT_CHECKPOINTS_DIR*. 
   ### 4.2 Evaluating baseline NMT system
   - Evaluate the baseline system with [this script](work/scripts/baseline/eval_inter_baseline.sh).
   - We need to evaluate BPEd test file with the best checkpoint. Use *--remove-bpe=sentencepiece* to remove the BPE segmentation from the output file. 
   - Use either Indic NLP or Moses to detokenize the output file as shown in the script.
   - This script calculates the final BLEU scores using SacreBLEU using the untouched test file of the target language.
## <a name="5"></a>5. XLM-R-fused NMT System
   ### 5.1 Training XLM-R-fused NMT systems
   - Train the XLM-R-fused systems with [this script](work/scripts/xlm-r-fused/train_xlm_r_fused.sh) which will use the system at [this location](work/systems/xlm-r-fused).
   - *BERT_NAME* stores the path to the XLM-R variant used with this system. We can use either pre-trained or finetuned variants here.
   - This script copies the best checkpoint from the baseline system and restores the parameters for the further training with the XLM-R-fused system.
   - This system was built over an earlier version of Fairseq which did not provide early stopping, so this script saves all the checkpoints for *--max-update* training steps, which are then evaluated later on.
   - For attention fusion at both the encoder and decoder side, use the *--arch* as *transformer_s2_iwslt_de_en* , while for the decoder-only fusion, use *transformer_iwslt_de_en*. 
   - Ensure to use a small learning rate, as the parameters are already near the optimum levels.
   ### 5.2 Evaluating XLM-R-fused NMT systems 
   - Evaluate the XLM-R-fused systems with [this script](work/scripts/xlm-r-fused/eval_inter_xlm_r_fused.sh), which is similar to the script used for the baseline system.
   - This script uses a particular checkpoint to compute BLEU score, while [another script](work/scripts/xlm-r-fused/eval_all_checkpoints_xlm_r_fused.sh) extends it to compute the scores for all the checkpoints. 
   - Like it was mentioned before, there is no early stopping feature with this system. So we saved all the checkpoints after some epochs and evaluated them with a validation set. Then, the best checkpoint can be set using BEST_CHECKPOINT in the former script for the test set evaluation. 
   - It uses path to two different test files which are used by different components of the XLM-R-fused NMT systems. *TEST_SRC_BPEd* points to the file which is used by the standard NMT-encoder, while *TEST_SRC_RAW* points to the raw source file which is used by the XLM-R component. Raw file is needed by the XLM_R as it uses its own internal tokenization using the tokenizer provided by the Huggingface transformers library. Ensure that *BERT_NAME* points to the corresponding XLM-R variant directory, so that it can access its corresponding Tokenizer.
## <a name="6"></a>6. Finetuning XLM-R
   ### 6.1 Multilingual and Monolingual Variants
   - We finetuned the XLM-R models to create the multilingual and monolingual variants of the original pre-trained models.    - Indo-Aryan-XLM-R-Base is the multilingual variant, which is created by finetuning XLM-R base with the related languages -- Hindi, Gujarati, Marathi, and Bengali. It exploits their syntactic, morphological, orthographic, and lexical similarities. 
   - Gujarati-XLM-R-Base and Gujarati-XLM-R-Large are the monolingual variants finetuned with the single Gujarati dataset. Further, Gujarati-Dev-XLM-R-Base is created with the Gujarati language converted to the Devanagari script. These models have been released at the HuggingFace hub which are available [here](hyperlink).
   - We used the Tensorflow variants of the XLM-R available [here](https://github.com/pytorch/fairseq/tree/master/examples/xlmr) as the pre-trained models.  
   ### 6.2 Preparing data
   - We primarily followed [this tutorial](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md), which we encourage you to visit before proceeding next.
   - Use [this script](work/scripts/finetune_xlm_r/prepare-bert-data-xlm-r.sh) to prepare the data. It prepares the training and validation files for a single monolingual dataset whose path is set using *RAW_MONO_SOURCE*. Then, it is preprocessed using the Indic NLP library, as done for the training files above. 
   - In this script, *BERT_NAME* points to the original pretrained XLM-R model which is used to access its dictionary.
   - Then, it is binarised using another variant of the Fairseq library available [here](work/systems/XLM-R-binariser/). It uses the XLM-R Tokenizer, which ensures that our data is tokenized in the same way as the original data was done for pretraining the XLM-R.
   - (Optional) If you want to finetune any other masked language model, the you need to import that language model in [this file](/work/systems/XLM-R-binariser/fairseq/fairseq/binarizer.py). Make the following changes:
      - Import the corresponding tokenizer instead of our default XLMRobertaTokenizer
		```
		from transformers import XLMRobertaTokenizer
		BertTokenizer = XLMRobertaTokenizer
		```
      - Replace the model name here
		```
		dict = BertTokenizer.from_pretrained("xlm-roberta-large")
		```
      - Replace the start and end tokens similar to a suggestion in the Preprocessing.
   - Similarly, prepare the data for other languages if you want to create a multilingual model.	
   ### 6.3 Training and Evaluating the Model
   - Use [this script](work/scripts/finetune_xlm_r/train-custom-bert-xlm-r_monolingual.sh) to finetune the monolingual variant.
   - It uses the same baseline NMT system which is used in Step 4.1 to finetune the model. Here, *RESTORE_POINT* points to the pre-trained model checkpoint. Use the *--task* as *masked_lm* for the monolingual variant. 
   - Similalrly, use [this script](work/scripts/finetune_xlm_r/train-custom-bert-xlm-r_indo-aryan.sh) to create the multilingual variant. It uses the *--task* as *multilingual_masked_lm* which merges the data from different languages.
   - It also resamples it to minimise the impact of data imbalance where larger datasets overpower the smaller ones. Use *--multilang-sampling-alpha* to adjust the sampling ratio. See the original XLM-R paper [[1]](#ref1) for the details. 
   - We saved the checkpoints at regular intervals, and picked the model with the minimum validation loss.
   ### 6.4 Converting Tensorflow Variant to PyTorch
   - We need to convert the saved checkpoint (in Tensorflow) to the PyTorch version. 
   - We assume that you have installed the transformers library in packages directory. Then, use the following command.
	```
	python packages/transformers/src/transformers/convert_roberta_original_pytorch_checkpoint_to_pytorch.py --roberta_checkpoint_path best_ck_dir/ --pytorch_dump_folder_path ./
	```
   - Here, *best_ck_dir* contains the Tensorflow version of the XLM-R checkpoint named as *model.pt*, *dict.txt* and sentencepiece model. Latter 2 files are available [here](Hyperlink). *pytorch_dump_folder_path* refers to the directory where the pytorch version needs to be saved.
   - Note that the Transformers library had some issues with the file *convert_roberta_original_pytorch_checkpoint_to_pytorch.py*, which we fixed and added to the the [utils directory](hyperlink). Replace this file and rebuild the library. 
   - (Optional) Use [HuggingFace Guide](hyperlink) directly to finetune the model without first converting to the Tensorflow variant. We found this approach extremely slow due to poor multi-GPU support provided by them. On the other hand, Fairseq has heavily optimized multi-GPU support, which helped us to finetune the models considerably faster. Read about these issues [here] (hyperlink).
   - After finetuning, just use the pytorch version to replace the original pre-trained models for training and evaluating the XLM-R-fused systems.
## <a name="7"></a>7. Script Conversion
## <a name="8"></a>8. Syntactic Analysis
## <a name="9"></a>9. Additional Info
## <a name="10"></a>10. References
<a id="ref1">[1]</a> [Conneau, Alexis, et al. "Unsupervised Cross-Lingual Representation Learning At Scale." arXiv preprint arXiv:1911.02116 (2019)](https://arxiv.org/pdf/1911.02116.pdf)

<a id="ref2">[2]</a> [Zhu, Jinhua, et al. "Incorporating BERT into Neural Machine Translation." International Conference on Learning Representations. 2019](https://openreview.net/pdf?id=Hyl7ygStwB)

<a id="ref3">[3]</a> [Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019, August). What Does BERT Look at? An Analysis of BERT’s Attention. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP (pp. 276-286).](https://www.aclweb.org/anthology/W19-4828.pdf)

<a id="ref4">[4]</a> [Anoop Kunchukuttan, Pratik Mehta, Pushpak Bhattacharyya. The IIT Bombay English-Hindi Parallel Corpus. Language Resources and Evaluation Conference. 2018. ](http://www.cfilt.iitb.ac.in/iitb_parallel/lrec2018_iitbparallel.pdf)
