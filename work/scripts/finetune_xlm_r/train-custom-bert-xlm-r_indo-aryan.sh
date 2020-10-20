set -x
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

HOME_DIR=/fs/bil0/atanwar
WORK_DIR=$HOME_DIR/work
PACKAGES_DIR=$HOME_DIR/packages

BERT_DIR=$WORK_DIR/bert
BERT_DATA_DIR=$BERT_DIR/data
BERT_MODELS_DIR=$BERT_DIR/models

SRC_LNG=indo-aryan
CUSTOM_BERT_MODEL_NAME_PATH=$BERT_MODELS_DIR/${SRC_LNG}-xlm-r
CUSTOM_BERT_DATA_DIR_GU=$BERT_DATA_DIR/gu
CUSTOM_BERT_DATA_DIR_HI=$BERT_DATA_DIR/hi
CUSTOM_BERT_DATA_DIR_BN=$BERT_DATA_DIR/bn
CUSTOM_BERT_DATA_DIR_MR=$BERT_DATA_DIR/mr
#BINARY_DATA_ALL_LANG=$CUSTOM_BERT_DATA_DIR_GU/binary:$CUSTOM_BERT_DATA_DIR_HI/binary:$CUSTOM_BERT_DATA_DIR_BN/binary:$CUSTOM_BERT_DATA_DIR_MR/binary

#BINARY_DATA_ALL_LANG=$CUSTOM_BERT_DATA_DIR_GU:$CUSTOM_BERT_DATA_DIR_HI:$CUSTOM_BERT_DATA_DIR_BN:$CUSTOM_BERT_DATA_DIR_MR

OUTPUT_DIR=$CUSTOM_BERT_MODEL_NAME_PATH
TOTAL_UPDATES=247536    # Total number of training steps
WARMUP_UPDATES=24753    # Warmup the learning rate over this many updates
PEAK_LR=2e-5          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=5        # Number of sequences per batch (batch size)
UPDATE_FREQ=13        # Increase the batch size 16x

#Baseline NMT-System
BASELINE_NMT_SYSTEM_DIR=$WORK_DIR/systems/baseline-NMT/fairseq
#RESTORE_POINT=$OUTPUT_DIR/checkpoint_last.pt
RESTORE_POINT=$BERT_MODELS_DIR/pre-trained/xlm-roberta/xlmr.base/model.pt

#Train baseline
python $BASELINE_NMT_SYSTEM_DIR/train.py $BERT_DATA_DIR \
    --task multilingual_masked_lm --criterion masked_lm --multilang-sampling-alpha 0.3 --memory-efficient-fp16 \
    --arch roberta_base --sample-break-mode eos --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --save-interval-updates 8251 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --skip-invalid-size-inputs-valid-test \
    --restore-file $RESTORE_POINT --save-dir $OUTPUT_DIR | tee -a $OUTPUT_DIR/training.log \
