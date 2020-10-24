set -x
set -e

export CUDA_VISIBLE_DEVICES=6

HOME_DIR=/fs/bil0/atanwar/repo/nmt-transfer-learning-xlm-r
WORK_DIR=$HOME_DIR/work
PACKAGES_DIR=$HOME_DIR/packages

BERT_DIR=$WORK_DIR/bert
BERT_DATA_DIR=$BERT_DIR/data
BERT_MODELS_DIR=$BERT_DIR/models

SRC_LNG=hi
CUSTOM_BERT_DATA_DIR=$BERT_DATA_DIR/$SRC_LNG
INTERMEDIATE_DIR=$CUSTOM_BERT_DATA_DIR/intermediate
BINARY_DIR=$CUSTOM_BERT_DATA_DIR/binary

TRAIN_FILE=$CUSTOM_BERT_DATA_DIR/bert_train.$SRC_LNG
VAL_FILE=$CUSTOM_BERT_DATA_DIR/bert_val.$SRC_LNG

if [ ! -d $CUSTOM_BERT_DATA_DIR ]
then
   mkdir $CUSTOM_BERT_DATA_DIR
   mkdir $INTERMEDIATE_DIR
fi

#Raw data
RAW_MONOLINGUAL_DATA_DIR=$WORK_DIR/raw-data/monolingual
RAW_MONO_SOURCE=$RAW_MONOLINGUAL_DATA_DIR/$SRC_LNG/${SRC_LNG}.mono
#RAW_MONO_SOURCE=$RAW_MONOLINGUAL_DATA_DIR/indo-aryan.mono

#IndicNLP
INDIC_NLP_DIR=$PACKAGES_DIR/indic-nlp
export INDIC_RESOURCES_PATH=$INDIC_NLP_DIR/indic_nlp_resources
INDIC_NLP_CLI_PARSER=$INDIC_NLP_DIR/indic_nlp_library/indicnlp/cli/cliparser.py

#move data to an intermediate dir
cp $RAW_MONO_SOURCE $INTERMEDIATE_DIR/bert_mono.${SRC_LNG}.raw

#pre-process
python $INDIC_NLP_CLI_PARSER normalize -l $SRC_LNG $INTERMEDIATE_DIR/bert_mono.${SRC_LNG}.raw $INTERMEDIATE_DIR/bert_mono.${SRC_LNG}.raw.norm
python $INDIC_NLP_CLI_PARSER tokenize -l $SRC_LNG $INTERMEDIATE_DIR/bert_mono.${SRC_LNG}.raw.norm $INTERMEDIATE_DIR/bert_mono.${SRC_LNG}.raw.norm.tok

#segregate into train and test.
shuf $INTERMEDIATE_DIR/bert_mono.${SRC_LNG}.raw.norm.tok \
  -o $INTERMEDIATE_DIR/bert_mono.${SRC_LNG}.raw.norm.tok.shuf
sed -n '1,50p' $INTERMEDIATE_DIR/bert_mono.${SRC_LNG}.raw.norm.tok.shuf > $VAL_FILE
sed -n '51,$p' $INTERMEDIATE_DIR/bert_mono.${SRC_LNG}.raw.norm.tok.shuf > $TRAIN_FILE
echo Done

#Bert saved models and cache
BERT_DIR=$WORK_DIR/bert
CACHE_DIR=$BERT_DIR/cache
BERT_MODELS_DIR=$BERT_DIR/models
BERT_TYPE=xlmr.base
BERT_NAME=$BERT_MODELS_DIR/pre-trained/xlm-roberta/$BERT_TYPE

#Prepare binary data for fairseq
BASELINE_NMT_SYSTEM_DIR=$WORK_DIR/systems/XLM-R-binariser/fairseq
#BERT_MODEL=xlm-roberta-large

python $BASELINE_NMT_SYSTEM_DIR/preprocess.py --only-source --srcdict $BERT_NAME/dict.txt \
 --trainpref $TRAIN_FILE \
 --validpref $VAL_FILE \
 --destdir $BINARY_DIR --workers 60 

