#!/usr/bin/env bash
# This script run train and test
usage() { echo "Usage: $0 [-dpr]" 1>&2; exit 1; }


# record execution time to log
time_log() {
    RUNNING_TIME=$(date +%T -d "1/1 + $SECONDS sec")
    echo -e "---------------------------------" | tee -a $LOG_FILE
    echo -e "$TYPE running time: $RUNNING_TIME" | tee -a $LOG_FILE
    let "TOTAL_SECONDS += $SECONDS"
}


mkdir -p log
LOG_FILE="log/run.log"
VERSION="v2.0.0"
echo "===============================" >> $LOG_FILE
echo "version: $VERSION" >> $LOG_FILE
TOTAL_SECONDS=0
# "CONFIG##*/" is the basename of CONFIG
while getopts "dpr" flag; do
  case "$flag" in
    d)
      SECONDS=0
      TYPE="debug"

      CONFIG="dataset_model"
      EXP="dataset_model"
      RUN_ID=$VERSION
      python3 mains/train/train.py -c "configs/$CONFIG.json" --run_id $RUN_ID --log_name "optuna.log" --optuna --name $EXP
      python3 mains/test/test.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --resume "saved/$EXP/$RUN_ID/tuned_model/model_best.pth" --run_id $RUN_ID

      time_log
      ;;
    p)
      SECONDS=0
      TYPE="preprocess"

      python3 preprocess.py

      time_log
      ;;
    r)
      SECONDS=0
      TYPE="run"

      CONFIG="dataset_model"
      EXP="dataset_model"
      RUN_ID=$VERSION

      time_log
      ;;
    *)
      usage
      ;;
  esac
done

TOTAL_TIME=$(date +%T -d "1/1 + $TOTAL_SECONDS sec")
echo -e "===============================" | tee -a $LOG_FILE
echo -e "total running time: $TOTAL_TIME" | tee -a $LOG_FILE
