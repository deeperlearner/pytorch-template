#!/bin/bash
# This script run train and test
# bash ./scripts/run/run.sh $1
# $1: debug or run_all

MODE=$1

if [ "$MODE" = "debug" ]; then
    CONFIG="dataset_model"
    EXP="dataset_model"
    RUN_ID="debug"
    python3 mains/train/train.py -c "configs/$CONFIG.json" --run_id $RUN_ID --log_name "optuna.log" --optuna --name $EXP
    python3 mains/test/test.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --resume "saved/$EXP/$RUN_ID/tuned_model/model_best.pth" --run_id $RUN_ID
else
    CONFIG="dataset_model"
    EXP="dataset_model"
    RUN_ID=0
fi
