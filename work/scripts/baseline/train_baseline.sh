export CUDA_VISIBLE_DEVICES=2,3
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

#Baseline NMT-System
BASELINE_NMT_SYSTEM_DIR=$WORK_DIR/systems/baseline-NMT/fairseq

#Baseline NMT Checkpoints dir
BASELINE_NMT_CHECKPOINTS_DIR=$PREPROCESSED_LANG_PAIR_DIR/baseline

#Train baseline
python $BASELINE_NMT_SYSTEM_DIR/train.py $BINARY_DATA \
 --weight-decay 0.0001 --clip-norm 0.1 --dropout 0.3 \
 --max-tokens 4000 \
 --optimizer adam --adam-betas '(0.9,0.98)' \
 --arch transformer_iwslt_de_en \
 --save-dir $BASELINE_NMT_CHECKPOINTS_DIR \
 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 --max-update 7400 --update-freq 8 --ddp-backend c10d \
 --source-lang $SRC_LNG --target-lang $TGT_LNG \
 --lr 0.001 --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 370 --warmup-init-lr '1e-07' \
 --validate-interval 1 --patience 10 --save-interval 2 --keep-interval-updates 10 \
 --share-all-embeddings \
 --task translation --eval-bleu \
 --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
 --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  | tee -a $BASELINE_NMT_CHECKPOINTS_DIR/training.log \
 
