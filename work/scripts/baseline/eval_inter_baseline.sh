export CUDA_VISIBLE_DEVICES=6

set -x
set -e

#NOTE:
#Using interactive.py requires BPEd data . 
#It does not shuffle the translated sentences. 
#Source test file should be tokenized and BPEd so that it can match the vocabulary of the model. Reference file should be detokenized and de-BPEd for explicitly calculating sacreBLEU (without using internal --sacrebleu)
#removebpe=sentencepiece should only be used if BPE was used with sentencepiece.

SRC_LNG=gu
TGT_LNG=hi
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

#Raw data(order of src and tgt should be according to raw data folder)
RAW_LANG_PAIR=$SRC_LNG-$TGT_LNG
RAW_DATA_DIR=$WORK_DIR/raw-data/parallel/original/$RAW_LANG_PAIR

#Preprocessed data
PREPROCESSED_LANG_PAIR_DIR=$WORK_DIR/preprocessed-data-and-models/$LANG_PAIR
PREPROCESSED_DATA_DIR=$PREPROCESSED_LANG_PAIR_DIR/preprocessed-data
TOK_BPE_DATA_DIR=$PREPROCESSED_DATA_DIR/tokenized-BPE
BINARY_DATA=$PREPROCESSED_DATA_DIR/binary
TEST_SRC_BPEd=$TOK_BPE_DATA_DIR/test.bpe.$LANG_PAIR.$SRC_LNG

#Baseline NMT-System
BASELINE_NMT_SYSTEM_DIR=$WORK_DIR/systems/baseline-NMT/fairseq/fairseq_cli

#Baseline NMT Checkpoints dir
BASELINE_NMT_CHECKPOINTS_DIR=$PREPROCESSED_LANG_PAIR_DIR/baseline
BEST_CHECKPOINT=$BASELINE_NMT_CHECKPOINTS_DIR/checkpoint_best.pt

#Log file
TGT_LOG=$BEST_CHECKPOINT.inter.log

#Evaluate baseline with interactive.py 
cat $TEST_SRC_BPEd | python $BASELINE_NMT_SYSTEM_DIR/interactive.py \
 $BINARY_DATA \
 -s $SRC_LNG -t $TGT_LNG \
 --path $BEST_CHECKPOINT \
 --batch-size 128 --beam 5 --lenpen 1.0 --buffer-size 1024 \
 --remove-bpe=sentencepiece --sacrebleu | tee $TGT_LOG

#Extracting all the hypothesis from the log. It corresponds to translated sentences.
grep ^H $TGT_LOG  | cut -f3- > $TGT_LOG.h

#Detokenizing sentences using Moses
#perl $DETOKENIZER -l $TGT_LNG < $TGT_LOG.h > $TGT_LOG.h.detok

#Detokenizing with IndicNLP
python $INDIC_NLP_CLI_PARSER detokenize -l $TGT_LNG $TGT_LOG.h $TGT_LOG.h.detok
 
#Compute sacreBLEU using original target side test data
cat $TGT_LOG.h.detok | sacrebleu $RAW_DATA_DIR/test.$RAW_LANG_PAIR.$TGT_LNG

