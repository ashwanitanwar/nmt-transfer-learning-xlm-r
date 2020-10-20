set -x
set -e

#This script will normalize and tokenize the data with IndicNLP/Moses/Sentencepiece. Then, it will segment to BPE using Sentencepiece. 

SRC_LNG=hi
TGT_LNG=mr
LANG_PAIR=$SRC_LNG-$TGT_LNG

HOME_DIR=/fs/bil0/atanwar
WORK_DIR=$HOME_DIR/work
PACKAGES_DIR=$HOME_DIR/packages

export PYTHONIOENCODING="UTF-8"

#Raw data(order of src and tgt should be according to raw data folder)
RAW_LANG_PAIR=$SRC_LNG-$TGT_LNG
#RAW_LANG_PAIR=$TGT_LNG-$SRC_LNG

#parallel data
RAW_DATA_DIR=$WORK_DIR/raw-data/parallel/$RAW_LANG_PAIR
RAW_TRAIN_SOURCE=$RAW_DATA_DIR/train.$RAW_LANG_PAIR.$SRC_LNG
RAW_TRAIN_TARGET=$RAW_DATA_DIR/train.$RAW_LANG_PAIR.$TGT_LNG
RAW_VAL_SOURCE=$RAW_DATA_DIR/dev.$RAW_LANG_PAIR.$SRC_LNG
RAW_VAL_TARGET=$RAW_DATA_DIR/dev.$RAW_LANG_PAIR.$TGT_LNG
RAW_TEST_SOURCE=$RAW_DATA_DIR/test.$RAW_LANG_PAIR.$SRC_LNG
RAW_TEST_TARGET=$RAW_DATA_DIR/test.$RAW_LANG_PAIR.$TGT_LNG

#monolingual data
RAW_MONOLINGUAL_DATA_DIR=$WORK_DIR/raw-data/monolingual
RAW_MONO_SOURCE=$RAW_MONOLINGUAL_DATA_DIR/$SRC_LNG/${SRC_LNG}.mono
RAW_MONO_TARGET=$RAW_MONOLINGUAL_DATA_DIR/$TGT_LNG/${TGT_LNG}.mono

#Preprocessed data
PREPROCESSED_LANG_PAIR_DIR=$WORK_DIR/preprocessed-data-and-models/$LANG_PAIR
PREPROCESSED_DATA_DIR=$PREPROCESSED_LANG_PAIR_DIR/preprocessed-data-syntactic
TOK_BPE_DATA_DIR=$PREPROCESSED_DATA_DIR/tokenized-BPE
BINARY_DATA=$PREPROCESSED_DATA_DIR/binary
INTERMEDIATE_DIR=$TOK_BPE_DATA_DIR/intermediate

#Intermediate Files after cleaning, lowercasing and tokenizing
INTERM_TRAIN_SOURCE=$INTERMEDIATE_DIR/train.interm.$LANG_PAIR.$SRC_LNG
INTERM_TRAIN_TARGET=$INTERMEDIATE_DIR/train.interm.$LANG_PAIR.$TGT_LNG
INTERM_VAL_SOURCE=$INTERMEDIATE_DIR/val.interm.$LANG_PAIR.$SRC_LNG
INTERM_VAL_TARGET=$INTERMEDIATE_DIR/val.interm.$LANG_PAIR.$TGT_LNG
INTERM_TEST_SOURCE=$INTERMEDIATE_DIR/test.interm.$LANG_PAIR.$SRC_LNG
INTERM_TEST_TARGET=$INTERMEDIATE_DIR/test.interm.$LANG_PAIR.$TGT_LNG
INTERM_MONO_SOURCE=$INTERMEDIATE_DIR/mono.interm.$LANG_PAIR.$SRC_LNG
INTERM_MONO_TARGET=$INTERMEDIATE_DIR/mono.interm.$LANG_PAIR.$TGT_LNG

#BPEd files.
TOK_BPE_TRAIN_SOURCE=$TOK_BPE_DATA_DIR/train.bpe.$LANG_PAIR.$SRC_LNG
TOK_BPE_TRAIN_TARGET=$TOK_BPE_DATA_DIR/train.bpe.$LANG_PAIR.$TGT_LNG
TOK_BPE_VAL_SOURCE=$TOK_BPE_DATA_DIR/val.bpe.$LANG_PAIR.$SRC_LNG
TOK_BPE_VAL_TARGET=$TOK_BPE_DATA_DIR/val.bpe.$LANG_PAIR.$TGT_LNG
TOK_BPE_TEST_SOURCE=$TOK_BPE_DATA_DIR/test.bpe.$LANG_PAIR.$SRC_LNG
TOK_BPE_TEST_TARGET=$TOK_BPE_DATA_DIR/test.bpe.$LANG_PAIR.$TGT_LNG

#Make these directories if they do not exist. 
if [ ! -d $PREPROCESSED_LANG_PAIR_DIR ]
then
   mkdir $PREPROCESSED_LANG_PAIR_DIR
   mkdir $PREPROCESSED_DATA_DIR
   mkdir $TOK_BPE_DATA_DIR
   mkdir $BINARY_DATA
   mkdir $INTERMEDIATE_DIR
fi

#Moses scripts
MOSES_DIR=$PACKAGES_DIR/mosesdecoder
MOSES_SCRIPTS=$MOSES_DIR/scripts
MOSES_TOKENIZER=$MOSES_SCRIPTS/tokenizer/tokenizer.perl
MOSES_DETOKENIZER=$MOSES_SCRIPTS/tokenizer/detokenizer.perl
MOSES_LC=$MOSES_SCRIPTS/tokenizer/lowercase.perl
MOSES_CLEAN=$MOSES_SCRIPTS/training/clean-corpus-n.perl
MOSES_NORMALIZER=$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl

#IndicNLP
INDIC_NLP_DIR=$PACKAGES_DIR/indic-nlp
export INDIC_RESOURCES_PATH=$INDIC_NLP_DIR/indic_nlp_resources
INDIC_NLP_CLI_PARSER=$INDIC_NLP_DIR/indic_nlp_library/indicnlp/cli/cliparser.py

clean_norm_tok() {
    local tokenizer=$1
    local input_file=$2
    local output_file=$3
    local lang=$4
    cp $input_file $output_file
    if [[ "$tokenizer" == "moses" ]]
    then
       echo Processing $input_file using MOSES tokenizer
       cat $output_file | $MOSES_NORMALIZER -l $lang > $output_file.norm
       cat $output_file.norm | $MOSES_TOKENIZER -l $lang > $output_file.norm.tok
    elif [[ "$tokenizer" == "indic-nlp" ]]
    then
       echo Processing $input_file using INDIC-NLP tokenizer
       python $INDIC_NLP_CLI_PARSER normalize -l $lang $output_file $output_file.norm
       python $INDIC_NLP_CLI_PARSER tokenize -l $lang $output_file.norm $output_file.norm.tok
    fi
    $MOSES_LC < $output_file.norm.tok > $output_file.norm.tok.lc
    cp $output_file.norm.tok.lc $output_file.done
    #print no. of lines after processing:
    local input_file_len=($(wc -l $input_file))
    local output_file_len=($(wc -l $output_file.done))
    echo "No. of sentences before and after pre-processing:" $input_file_len "and" $output_file_len "for" $input_file
}

<<'CT'
clean_norm_tok indic-nlp $RAW_TRAIN_SOURCE $INTERM_TRAIN_SOURCE $SRC_LNG
clean_norm_tok indic-nlp $RAW_TRAIN_TARGET $INTERM_TRAIN_TARGET $TGT_LNG
clean_norm_tok indic-nlp $RAW_VAL_SOURCE $INTERM_VAL_SOURCE $SRC_LNG
clean_norm_tok indic-nlp $RAW_VAL_TARGET $INTERM_VAL_TARGET $TGT_LNG
clean_norm_tok indic-nlp $RAW_TEST_SOURCE $INTERM_TEST_SOURCE $SRC_LNG
clean_norm_tok indic-nlp $RAW_TEST_TARGET $INTERM_TEST_TARGET $TGT_LNG
clean_norm_tok indic-nlp $RAW_MONO_SOURCE $INTERM_MONO_SOURCE $SRC_LNG
clean_norm_tok indic-nlp $RAW_MONO_TARGET $INTERM_MONO_TARGET $TGT_LNG

#clean_norm_tok moses $RAW_TRAIN_SOURCE $INTERM_TRAIN_SOURCE $SRC_LNG
#clean_norm_tok moses $RAW_TRAIN_TARGET $INTERM_TRAIN_TARGET $TGT_LNG
#clean_norm_tok moses $RAW_VAL_SOURCE $INTERM_VAL_SOURCE $SRC_LNG
#clean_norm_tok moses $RAW_VAL_TARGET $INTERM_VAL_TARGET $TGT_LNG
#clean_norm_tok moses $RAW_TEST_SOURCE $INTERM_TEST_SOURCE $SRC_LNG
#clean_norm_tok moses $RAW_TEST_TARGET $INTERM_TEST_TARGET $TGT_LNG
#clean_norm_tok moses $RAW_MONO_SOURCE $INTERM_MONO_SOURCE $SRC_LNG
#clean_norm_tok moses $RAW_MONO_TARGET $INTERM_MONO_TARGET $TGT_LNG
#################BPE Processing#########################
#mix and shuffle mono and train parallel for SP learning
MONO_SOURCE_LEN=($(wc -l $INTERM_MONO_SOURCE.done))
MONO_TARGET_LEN=($(wc -l $INTERM_MONO_TARGET.done))
MIN_LENGTH=$(( $MONO_SOURCE_LEN < $MONO_TARGET_LEN ? $MONO_SOURCE_LEN : $MONO_TARGET_LEN ))

#Draw MIN_LENGTH no. of lines from both the monolingual datasets.
shuf -n $MIN_LENGTH $INTERM_MONO_SOURCE.done -o $INTERM_MONO_SOURCE.done.$MIN_LENGTH
shuf -n $MIN_LENGTH $INTERM_MONO_TARGET.done -o $INTERM_MONO_TARGET.done.$MIN_LENGTH

cat $INTERM_TRAIN_SOURCE.done $INTERM_MONO_SOURCE.done.$MIN_LENGTH | shuf -o \
 $INTERMEDIATE_DIR/mono-train.shuf.$SRC_LNG
cat $INTERM_TRAIN_TARGET.done $INTERM_MONO_TARGET.done.$MIN_LENGTH | shuf -o \
 $INTERMEDIATE_DIR/mono-train.shuf.$TGT_LNG
cat $INTERMEDIATE_DIR/mono-train.shuf.$SRC_LNG $INTERMEDIATE_DIR/mono-train.shuf.$TGT_LNG | shuf -o \
 $INTERMEDIATE_DIR/mono-train.shuf.shared 

#learn spm and produce codes using joint data
VOCAB_SIZE=32000
CHAR_COV=0.9995
spm_train --input=$INTERMEDIATE_DIR/mono-train.shuf.shared \
 --model_prefix=$TOK_BPE_DATA_DIR/sentencepiece.$LANG_PAIR \
 --vocab_size=$VOCAB_SIZE \
 --character_coverage=$CHAR_COV --model_type=bpe

#Extract their vocabulary
spm_encode --model=$TOK_BPE_DATA_DIR/sentencepiece.$LANG_PAIR.model \
 --generate_vocabulary < $INTERMEDIATE_DIR/mono-train.shuf.$SRC_LNG > $TOK_BPE_DATA_DIR/vocab.$SRC_LNG
spm_encode --model=$TOK_BPE_DATA_DIR/sentencepiece.$LANG_PAIR.model \
 --generate_vocabulary < $INTERMEDIATE_DIR/mono-train.shuf.$TGT_LNG > $TOK_BPE_DATA_DIR/vocab.$TGT_LNG
CT
#Encode train, val and test files with corresponding vocab
spm_encode --model=$TOK_BPE_DATA_DIR/sentencepiece.$LANG_PAIR.model \
 --vocabulary=$TOK_BPE_DATA_DIR/vocab.$SRC_LNG --vocabulary_threshold=50 \
 --output_format=piece < $INTERM_TRAIN_SOURCE.done > $TOK_BPE_TRAIN_SOURCE
<<'CT'
spm_encode --model=$TOK_BPE_DATA_DIR/sentencepiece.$LANG_PAIR.model \
 --vocabulary=$TOK_BPE_DATA_DIR/vocab.$TGT_LNG --vocabulary_threshold=50 \
 --output_format=piece  < $INTERM_TRAIN_TARGET.done > $TOK_BPE_TRAIN_TARGET
CT
spm_encode --model=$TOK_BPE_DATA_DIR/sentencepiece.$LANG_PAIR.model \
 --vocabulary=$TOK_BPE_DATA_DIR/vocab.$SRC_LNG --vocabulary_threshold=50 \
 --output_format=piece  < $INTERM_VAL_SOURCE.done > $TOK_BPE_VAL_SOURCE
<<'CT'
spm_encode --model=$TOK_BPE_DATA_DIR/sentencepiece.$LANG_PAIR.model \
 --vocabulary=$TOK_BPE_DATA_DIR/vocab.$TGT_LNG --vocabulary_threshold=50 \
 --output_format=piece  < $INTERM_VAL_TARGET.done > $TOK_BPE_VAL_TARGET
spm_encode --model=$TOK_BPE_DATA_DIR/sentencepiece.$LANG_PAIR.model \
 --vocabulary=$TOK_BPE_DATA_DIR/vocab.$SRC_LNG --vocabulary_threshold=50 \
 --output_format=piece  < $INTERM_TEST_SOURCE.done > $TOK_BPE_TEST_SOURCE
spm_encode --model=$TOK_BPE_DATA_DIR/sentencepiece.$LANG_PAIR.model \
 --vocabulary=$TOK_BPE_DATA_DIR/vocab.$TGT_LNG --vocabulary_threshold=50 \
 --output_format=piece  < $INTERM_TEST_TARGET.done > $TOK_BPE_TEST_TARGET
CT

#prepare for BERT
cp $INTERM_TRAIN_SOURCE.done $TOK_BPE_DATA_DIR/train.bpe.$LANG_PAIR.bert.$SRC_LNG 
cp $INTERM_VAL_SOURCE.done $TOK_BPE_DATA_DIR/val.bpe.$LANG_PAIR.bert.$SRC_LNG
#cp $INTERM_TEST_SOURCE.done $TOK_BPE_DATA_DIR/test.bpe.$LANG_PAIR.bert.$SRC_LNG

#Prepare binary data for fairseq
BERT_FUSED_NMT_SYSTEM_DIR=$WORK_DIR/systems/xlm-r-fused/bert-nmt
#Bert saved models and cache
BERT_DIR=$WORK_DIR/bert
BERT_MODELS_DIR=$BERT_DIR/models
BERT_TYPE=xlmr.base
BERT_NAME=$BERT_MODELS_DIR/pre-trained/xlm-roberta/$BERT_TYPE

SRCDICT=$PREPROCESSED_DATA_DIR/binary/dict.hi.txt

python $BERT_FUSED_NMT_SYSTEM_DIR/preprocess.py --only-source --srcdict $SRCDICT  \
 --source-lang $SRC_LNG \
 --trainpref $TOK_BPE_DATA_DIR/train.bpe.$LANG_PAIR \
 --validpref $TOK_BPE_DATA_DIR/val.bpe.$LANG_PAIR \
 --destdir $BINARY_DATA --joined-dictionary --bert-model-name $BERT_NAME --workers 60

