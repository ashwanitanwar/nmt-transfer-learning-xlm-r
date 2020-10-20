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

TEST_OR_TRAIN=train
ATTN_SELF_OR_BERT=self

export PYTHONIOENCODING="UTF-8"
<<'CT'
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
CT
#Preprocessed data
PREPROCESSED_LANG_PAIR_DIR=$WORK_DIR/preprocessed-data-and-models/$LANG_PAIR
PREPROCESSED_DATA_DIR=$PREPROCESSED_LANG_PAIR_DIR/preprocessed-data-syntactic
TOK_BPE_DATA_DIR=$PREPROCESSED_DATA_DIR/tokenized-BPE
BINARY_DATA=$PREPROCESSED_DATA_DIR/binary
TEST_SRC_BPEd=$TOK_BPE_DATA_DIR/${TEST_OR_TRAIN}.bpe.$LANG_PAIR.$SRC_LNG
<<'CT'
#Raw data(order of src and tgt should be according to raw data folder)
RAW_LANG_PAIR=$SRC_LNG-$TGT_LNG
RAW_DATA_DIR=$WORK_DIR/raw-data/parallel/$RAW_LANG_PAIR
#TEST_SRC_RAW=$RAW_DATA_DIR/test.$RAW_LANG_PAIR.$SRC_LNG
TEST_SRC_RAW=$TOK_BPE_DATA_DIR/train.bpe.$LANG_PAIR.bert.$SRC_LNG
CT
#BERT_fused NMT-System
#BERT_FUSED_NMT_SYSTEM_DIR=$WORK_DIR/systems/xlm-r-fused-extract-attn/bert-nmt

#Bert saved models and cache
BERT_DIR=$WORK_DIR/bert
CACHE_DIR=$BERT_DIR/cache
BERT_MODELS_DIR=$BERT_DIR/models
#BERT_TYPE=gu-xlm-r
BERT_TYPE=xlmr.large
BERT_NAME=$BERT_MODELS_DIR/pre-trained/xlm-roberta/$BERT_TYPE

#Bert-fused NMT Checkpoints dir
#BERT_FOLDER=${BERT_TYPE}-fused-S2-encoder
BERT_FOLDER=baseline
BERT_FUSED_NMT_CHECKPOINTS_DIR=$PREPROCESSED_LANG_PAIR_DIR/$BERT_FOLDER
#BEST_CHECKPOINT=$BERT_FUSED_NMT_CHECKPOINTS_DIR/checkpoint194.pt

ATTN_ANALYSIS_DIR=$WORK_DIR/syntactic_analysis/attention-analysis
outpath=$ATTN_ANALYSIS_DIR/data/processed/hi/saved_attn_pkl/$BERT_FOLDER/$ATTN_SELF_OR_BERT/$TEST_OR_TRAIN 
mkdir -p $outpath

if [[ $ATTN_SELF_OR_BERT == 'bert' ]]
then
    echo --bert_attn enabled
    if_bert='--bert_attn'
elif [[ $ATTN_SELF_OR_BERT == 'self' ]]
then
    echo --bert_attn disabled
    if_bert=''
fi

python $ATTN_ANALYSIS_DIR/extract_attention.py \
--preprocessed-data-file  $ATTN_ANALYSIS_DIR/data/processed/hi/${TEST_OR_TRAIN}.json \
--batch_size 64 --word_level --attn_maps_dir $BERT_FUSED_NMT_CHECKPOINTS_DIR/attn/$TEST_OR_TRAIN \
--self_attn_sentencepiece $TOK_BPE_DATA_DIR/sentencepiece.$LANG_PAIR.model \
--self_attn_vocab $TOK_BPE_DATA_DIR/vocab.$SRC_LNG \
--outpath $outpath --bert-dir $BERT_NAME \
$if_bert

<<'CT'
#Log file
TGT_LOG=$BEST_CHECKPOINT.inter.log

#Concat BPEd and raw file for interactive.py
#paste -d "\n" $TEST_SRC_BPEd $TEST_SRC_RAW > $TOK_BPE_DATA_DIR/train.concat.$LANG_PAIR.$SRC_LNG

#Evaluate bert-fused with interactive.py 
python $BERT_FUSED_NMT_SYSTEM_DIR/interactive.py \
 $BINARY_DATA \
 -s $SRC_LNG -t $TGT_LNG --input $TOK_BPE_DATA_DIR/train.concat.$LANG_PAIR.$SRC_LNG \
 --path $BEST_CHECKPOINT --bert-model-name $BERT_NAME --save_attn_maps $BERT_FUSED_NMT_CHECKPOINTS_DIR/attn/train \
 --batch-size 64 --beam 5 --lenpen 1.0 --buffer-size 1024 \
 --remove-bpe=sentencepiece --sacrebleu | tee $TGT_LOG

#Extracting all the hypothesis from the log. It corresponds to translated sentences.
grep ^H $TGT_LOG  | cut -f3- > $TGT_LOG.h

#Detokenizing sentences using Moses( may use some other detokenizer such as IndicNLP)
#perl $DETOKENIZER -l $TGT_LNG < $TGT_LOG.h > $TGT_LOG.h.detok
python $INDIC_NLP_CLI_PARSER detokenize -l $TGT_LNG $TGT_LOG.h $TGT_LOG.h.detok

#Compute sacreBLEU using original target side test data
cat $TGT_LOG.h.detok | sacrebleu $RAW_DATA_DIR/train.$RAW_LANG_PAIR.$TGT_LNG
CT
