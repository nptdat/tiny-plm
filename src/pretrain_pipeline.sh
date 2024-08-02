#!/bin/bash

set -e

COMMAND=$1

if [ -z $COMMAND ]; then
    COMMAND="all"
fi

echo "COMMAND: ${COMMAND}"
echo "DEVICE: ${DEVICE}"
echo "DATA_PATH: ${DATA_PATH}"
echo "TRAIN_PARAM_JSONS: ${TRAIN_PARAM_JSONS}"


print_title () {
    echo "############################"
    echo $1
    echo "############################"
}


############################
# Preparation
############################
mkdir -p ${DATA_PATH}

############################
# Download data
############################
if [ $COMMAND == "download-cc100-ja" ] || [ $COMMAND == "all" ]; then
    print_title "----- Downloading cc100-ja raw data from ${CC100_JA_URL}..."
    wget ${CC100_JA_URL} -O ${CC100_JA_RAW_FILE}
fi

if [ $COMMAND == "download-jawiki" ] || [ $COMMAND == "all" ]; then
    print_title "----- Downloading jawiki raw data ${JAWIKI_URL}..."
    wget ${JAWIKI_URL} -O ${JAWIKI_RAW_FILE}
fi

############################
# Extract text data
############################
if [ $COMMAND == "build-dataset-cc100-ja" ] || [ $COMMAND == "all" ]; then
    print_title "----- Extracting text from raw data from cc100-ja..."
    python src/build_corpus_cc100_ja.py \
        ${CC100_JA_RAW_FILE} \
        ${CC100_JA_PATH} \
        --num-files ${CC100_NUM_FILES} \
        --max-sentence-num ${CC100_MAX_SENTENCE_NUM} \
        --max-validation-sentence-num ${CC100_MAX_VALIDATION_SENTENCE_NUM}
fi

if [ $COMMAND == "build-dataset-jawiki" ] || [ $COMMAND == "all" ]; then
    print_title "----- Extracting text from raw data from jawiki..."
    python src/build_corpus_wiki.py \
        ${JAWIKI_RAW_FILE} \
        ${JAWIKI_PATH} \
        --n-workers ${JAWIKI_NUM_WORKERS} \
        --max-document-num ${JAWIKI_MAX_DOCUMENTS}
fi


if [ $COMMAND == "train-tokenizer" ] || [ $COMMAND == "all" ]; then
    ############################
    # Random some lines to train tokenizer
    ############################
    print_title "----- Random some lines to train a tokenizer..."
    cat ${JAWIKI_PATH}/train/*.* | grep -a -v '^$' | shuf | head -n ${TOKENIZER_INPUT_SENTENCES} > ${TOKENIZER_INPUT_FILE}

    ############################
    # Train the tokenizer
    ############################
    print_title "----- Training the tokenizer..."
    python src/train_tokenizer.py \
        ${TOKENIZER_INPUT_FILE} \
        ${TOKENIZER_OUTPUT_PATH}
fi


if [ $COMMAND == "train-model" ] || [ $COMMAND == "all" ]; then
    ############################
    # Train BERT model
    ############################
    print_title "----- Pre-train model with multi-phrases..."
    IFS=':' list=($TRAIN_PARAM_JSONS)
    for param_file in "${list[@]}"; do
        print_title "----- Train the model with json_file=${param_file}..."
        # if ! deepspeed --num_gpus=1 src/run_mlm.py ${TRAIN_PARAM_JSON}; then
        if ! python src/run_mlm.py ${param_file}; then
            echo "[ERROR] BERT pretraining failed with ${param_file}"
            exit 1
        fi
    done

    echo "-- Finished BERT pretraining!"
fi
