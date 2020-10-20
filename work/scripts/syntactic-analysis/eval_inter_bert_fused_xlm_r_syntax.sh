export CUDA_VISIBLE_DEVICES=6

set -x
set -e

#NOTE:
#Using interactive.py requires raw BPEd data . BERT-fused version requires concat of BPEd and non-BPEd raw data(to feed in BERT) 
#It does not shuffle the translated sentences. 
#Source test file should be tokenized and BPEd so that it can match the vocabulary of the model. Reference file should be detokenized and de-BPEd for explicitly calculating sacreBLEU (without using internal --sacrebleu)
#removebpe=sentencepiece should only be used if BPE was used with sentencepiece.

SRC_LNG=hi
TGT_LNG=mr
LANG_PAIR=$SRC_LNG-$TGT_LNG

HOME_DIR=/fs/bil0/atanwar
WORK_DIR=$HOME_DIR/work
PACKAGES_DIR=$HOME_DIR/packages

export PYTHONIOENCODING="UTF-8"

#Moses scripts; We just use detokenizer here. 
MOSES_DIR=$PACKAGES_DIR/mosesdecoder
SCRIPTS=$MOSES_DIR/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

#IndicNLP
INDIC_NLP_DIR=$PACKAGES_DIR/indic-nlp
export INDIC_RESOURCES_PATH=$INDIC_NLP_DIR/indic_nlp_resources
INDIC_NLP_CLI_PARSER=$INDIC_NLP_DIR/indic_nlp_library/indicnlp/cli/cliparser.py

#Preprocessed data
PREPROCESSED_LANG_PAIR_DIR=$WORK_DIR/preprocessed-data-and-models/$LANG_PAIR
PREPROCESSED_DATA_DIR=$PREPROCESSED_LANG_PAIR_DIR/preprocessed-data-syntactic
TOK_BPE_DATA_DIR=$PREPROCESSED_DATA_DIR/tokenized-BPE
BINARY_DATA=$PREPROCESSED_DATA_DIR/binary
TEST_SRC_BPEd=$TOK_BPE_DATA_DIR/train.bpe.$LANG_PAIR.$SRC_LNG

#Raw data(order of src and tgt should be according to raw data folder)
RAW_LANG_PAIR=$SRC_LNG-$TGT_LNG
RAW_DATA_DIR=$WORK_DIR/raw-data/parallel/$RAW_LANG_PAIR
#TEST_SRC_RAW=$RAW_DATA_DIR/test.$RAW_LANG_PAIR.$SRC_LNG
TEST_SRC_RAW=$TOK_BPE_DATA_DIR/train.bpe.$LANG_PAIR.bert.$SRC_LNG

#BERT_fused NMT-System
BERT_FUSED_NMT_SYSTEM_DIR=$WORK_DIR/systems/xlm-r-fused-extract-attn/bert-nmt

#Bert saved models and cache
BERT_DIR=$WORK_DIR/bert
CACHE_DIR=$BERT_DIR/cache
BERT_MODELS_DIR=$BERT_DIR/models
#BERT_TYPE=gu-xlm-r
BERT_TYPE=indo-aryan-xlm-r
BERT_NAME=$BERT_MODELS_DIR/$BERT_TYPE

#Bert-fused NMT Checkpoints dir
BERT_FOLDER=${BERT_TYPE}-fused-S2-encoder
BERT_FUSED_NMT_CHECKPOINTS_DIR=$PREPROCESSED_LANG_PAIR_DIR/$BERT_FOLDER
BEST_CHECKPOINT=$BERT_FUSED_NMT_CHECKPOINTS_DIR/checkpoint194.pt

#Log file
TGT_LOG=$BEST_CHECKPOINT.inter.log

#Concat BPEd and raw file for interactive.py
paste -d "\n" $TEST_SRC_BPEd $TEST_SRC_RAW > $TOK_BPE_DATA_DIR/train.concat.$LANG_PAIR.$SRC_LNG

#Evaluate bert-fused with interactive.py 
python $BERT_FUSED_NMT_SYSTEM_DIR/interactive.py \
 $BINARY_DATA \
 -s $SRC_LNG -t $TGT_LNG --input $TOK_BPE_DATA_DIR/train.concat.$LANG_PAIR.$SRC_LNG \
 --path $BEST_CHECKPOINT --bert-model-name $BERT_NAME --save_attn_maps $BERT_FUSED_NMT_CHECKPOINTS_DIR/attn/train \
 --batch-size 64 --beam 5 --lenpen 1.0 --buffer-size 1024 \
 --remove-bpe=sentencepiece --sacrebleu | tee $TGT_LOG
<<'CT'
#Extracting all the hypothesis from the log. It corresponds to translated sentences.
grep ^H $TGT_LOG  | cut -f3- > $TGT_LOG.h

#Detokenizing sentences using Moses( may use some other detokenizer such as IndicNLP)
#perl $DETOKENIZER -l $TGT_LNG < $TGT_LOG.h > $TGT_LOG.h.detok
python $INDIC_NLP_CLI_PARSER detokenize -l $TGT_LNG $TGT_LOG.h $TGT_LOG.h.detok

#Compute sacreBLEU using original target side test data
cat $TGT_LOG.h.detok | sacrebleu $RAW_DATA_DIR/train.$RAW_LANG_PAIR.$TGT_LNG
CT
