export CUDA_VISIBLE_DEVICES=1

set -x
set -e

#NOTE:
#Using interactive.py requires raw BPEd data . BERT-fused version requires concat of BPEd and non-BPEd raw data(to feed in BERT) 
#It does not shuffle the translated sentences. 
#Source test file should be tokenized and BPEd so that it can match the vocabulary of the model. Reference file should be detokenized and de-BPEd for explicitly calculating sacreBLEU (without using internal --sacrebleu)
#removebpe=sentencepiece should only be used if BPE was used with sentencepiece.

SRC_LNG=gu
TGT_LNG=hi
LANG_PAIR=$SRC_LNG-$TGT_LNG
TEST_TYPE=test

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
RAW_DATA_DIR=$WORK_DIR/raw-data/parallel/augmented/$RAW_LANG_PAIR

#Preprocessed data
PREPROCESSED_LANG_PAIR_DIR=$WORK_DIR/preprocessed-data-and-models/$LANG_PAIR-script-converted-only-bert
PREPROCESSED_DATA_DIR=$PREPROCESSED_LANG_PAIR_DIR/preprocessed-data
TOK_BPE_DATA_DIR=$PREPROCESSED_DATA_DIR/tokenized-BPE
BINARY_DATA=$PREPROCESSED_DATA_DIR/binary
VAL_SRC_BPEd=$TOK_BPE_DATA_DIR/$TEST_TYPE.bpe.$LANG_PAIR.$SRC_LNG
VAL_SRC_RAW=$TOK_BPE_DATA_DIR/$TEST_TYPE.bpe.$LANG_PAIR.bert.$SRC_LNG

#BERT_fused NMT-System
BERT_FUSED_NMT_SYSTEM_DIR=$WORK_DIR/systems/xlm-r-fused/bert-nmt

#Bert saved models and cache
BERT_DIR=$WORK_DIR/bert
CACHE_DIR=$BERT_DIR/cache
BERT_MODELS_DIR=$BERT_DIR/models
BERT_TYPE=xlmr.base
BERT_NAME=$BERT_MODELS_DIR/pre-trained/xlm-roberta/$BERT_TYPE

#Bert-fused NMT Checkpoints dir
BERT_FOLDER=${BERT_TYPE}-fused
BERT_FUSED_NMT_CHECKPOINTS_DIR=$PREPROCESSED_LANG_PAIR_DIR/$BERT_FOLDER
OUTPUT_VAL_BLEU=$BERT_FUSED_NMT_CHECKPOINTS_DIR/all_checkpoints_${TEST_TYPE}_bleu.txt

#echo "Validation BLEU for all the checkpoints for" $LANG_PAIR $BERT_NAME > $OUTPUT_VAL_BLEU
#Concat BPEd and raw file for interactive.py
paste -d "\n" $VAL_SRC_BPEd $VAL_SRC_RAW > $TOK_BPE_DATA_DIR/$TEST_TYPE.concat.$LANG_PAIR.$SRC_LNG

for BEST_CHECKPOINT in $(ls $BERT_FUSED_NMT_CHECKPOINTS_DIR/checkpoint[0-9]*.pt); do
  #Log file
  TGT_LOG=$BEST_CHECKPOINT.$TEST_TYPE.inter.log

  #Evaluate bert-fused with interactive.py 
  cat $TOK_BPE_DATA_DIR/$TEST_TYPE.concat.$LANG_PAIR.$SRC_LNG | python $BERT_FUSED_NMT_SYSTEM_DIR/interactive.py \
   $BINARY_DATA \
   -s $SRC_LNG -t $TGT_LNG \
   --path $BEST_CHECKPOINT --bert-model-name $BERT_NAME \
   --batch-size 128 --beam 5 --lenpen 1.0 --buffer-size 1024 \
   --remove-bpe=sentencepiece --sacrebleu | tee $TGT_LOG

  #Extracting all the hypothesis from the log. It corresponds to translated sentences.
  grep ^H $TGT_LOG  | cut -f3- > $TGT_LOG.h

  #Detokenizing sentences using Moses( may use some other detokenizer such as IndicNLP)
  #perl $DETOKENIZER -l $TGT_LNG < $TGT_LOG.h > $TGT_LOG.h.detok
  python $INDIC_NLP_CLI_PARSER detokenize -l $SRC_LNG $TGT_LOG.h $TGT_LOG.h.detok

  #python $INDIC_NLP_CLI_PARSER script_convert -s $SRC_LNG -t $TGT_LNG $TGT_LOG.h.detok $TGT_LOG.h.detok.sc

  echo $BEST_CHECKPOINT >> $OUTPUT_VAL_BLEU
  #Compute sacreBLEU using original target side test data
  cat $TGT_LOG.h.detok | sacrebleu $RAW_DATA_DIR/$TEST_TYPE.$RAW_LANG_PAIR.$TGT_LNG --short >> $OUTPUT_VAL_BLEU
done

sort -g $OUTPUT_VAL_BLEU
