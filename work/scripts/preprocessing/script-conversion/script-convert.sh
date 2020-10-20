PACKAGES_DIR=/fs/bil0/atanwar/packages

INDIC_NLP_DIR=$PACKAGES_DIR/indic-nlp
export INDIC_RESOURCES_PATH=$INDIC_NLP_DIR/indic_nlp_resources
INDIC_NLP_CLI_PARSER=$INDIC_NLP_DIR/indic_nlp_library/indicnlp/cli/cliparser.py

SRC_LNG=$1
TGT_LNG=$2
Input=$3
Output=$4
python $INDIC_NLP_CLI_PARSER script_convert -s $SRC_LNG -t $TGT_LNG $Input $Output
