#!/bin/bash
# This script run train and test
# bash ./scripts/run/run.sh $1
# $1: debug or run_all

MODE=$1

if [ "$MODE" = "debug" ]; then
    CONFIG="dataset_model"
    EXP="dataset_model"
    RUN_ID="debug"
    python3 mains/single_process/train.py -c "configs/$CONFIG.json" --run_id $RUN_ID
    python3 mains/single_process/test.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID
else
    CONFIG="dataset_model"
    EXP="dataset_model"
    RUN_ID=0

    CONFIG="dataset_model"
    EXP="dataset_model"
    RUN_ID=0
fi
