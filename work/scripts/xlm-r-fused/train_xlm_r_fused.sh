export CUDA_VISIBLE_DEVICES=6
set -x
set -e

HOME_DIR=/fs/bil0/atanwar/repo/nmt-transfer-learning-xlm-r
WORK_DIR=$HOME_DIR/work

SRC_LNG=en
TGT_LNG=hi
LANG_PAIR=$SRC_LNG-$TGT_LNG

#Preprocessed data
PREPROCESSED_LANG_PAIR_DIR=$WORK_DIR/preprocessed-data-and-models/$LANG_PAIR
PREPROCESSED_DATA_DIR=$PREPROCESSED_LANG_PAIR_DIR/preprocessed-data
TOK_BPE_DATA_DIR=$PREPROCESSED_DATA_DIR/tokenized-BPE
BINARY_DATA=$PREPROCESSED_DATA_DIR/binary

#Bert-fused NMT-System
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

if [ ! -d $BERT_FUSED_NMT_CHECKPOINTS_DIR ]
then 
    mkdir $BERT_FUSED_NMT_CHECKPOINTS_DIR
fi

#Baseline NMT Checkpoints dir
BASELINE_NMT_CHECKPOINTS_DIR=$PREPROCESSED_LANG_PAIR_DIR/baseline

#move best baseline checkpoint to BERT-fused checkpoints directory
if [ ! -f $BERT_FUSED_NMT_CHECKPOINTS_DIR/checkpoint_nmt.pt ]
then
    cp $BASELINE_NMT_CHECKPOINTS_DIR/checkpoint_best.pt $BERT_FUSED_NMT_CHECKPOINTS_DIR/checkpoint_nmt.pt
fi
if [ ! -f "$BERT_FUSED_NMT_CHECKPOINTS_DIR/checkpoint_last.pt" ]
then
warmup="--warmup-from-nmt --reset-lr-scheduler"
else
warmup=""
fi

#Train bert-fused NMT
python $BERT_FUSED_NMT_SYSTEM_DIR/train.py $BINARY_DATA \
 --source-lang $SRC_LNG --target-lang $TGT_LNG \
 --weight-decay 0.0001 --dropout 0.3 \
 --max-tokens 4000 --update-freq 8 \
 --optimizer adam --adam-betas '(0.9,0.98)' \
 --arch transformer_iwslt_de_en \
 --save-dir $BERT_FUSED_NMT_CHECKPOINTS_DIR \
 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 --max-update 7650 \
 --lr 0.0005 --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 510 \
 --warmup-init-lr '1e-07' $warmup \
 --bert-model-name $BERT_NAME \
 --encoder-bert-dropout --encoder-bert-dropout-ratio 0.5 \
 --save-interval 2 \
 --share-all-embeddings | tee -a $BERT_FUSED_NMT_CHECKPOINTS_DIR/training.log
