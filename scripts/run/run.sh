#!/usr/bin/env bash
# This script run train and test
usage() { echo "Usage: $0 [-dpr]" 1>&2; exit 1; }


# "CONFIG##*/" is the basename of CONFIG
while getopts "dpr" flag; do
  case "$flag" in
    d)  # debug
      CONFIG="dataset_model"
      EXP="dataset_model"
      RUN_ID="debug"
      python3 mains/train/train.py -c "configs/$CONFIG.json" --run_id $RUN_ID --log_name "optuna.log" --optuna --name $EXP
      python3 mains/test/test.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --resume "saved/$EXP/$RUN_ID/tuned_model/model_best.pth" --run_id $RUN_ID
      ;;
    p)  # preprocess
      python3 preprocess.py
      ;;
    r)  # run all
      CONFIG="dataset_model"
      EXP="dataset_model"
      RUN_ID=0
      ;;
    *)
      usage
      ;;
  esac
done
