#!/usr/bin/env bash
# ------------------
#  PyTorch Template
# ------------------
# Repository    : https://github.com/deeperlearner/pytorch-template
VERSION="v1.0.0"


# This script run train and test
usage() { echo "Usage: $0 [-abc]" 1>&2; exit 1; }

# record execution time to log
time_log() {
    RUNNING_TIME=$(($SECONDS/86400))" days, "$(date +%T -d "1/1 + $SECONDS sec")
    echo -e "---------------------------------" | tee -a $LOG_FILE
    echo -e "$TYPE running time: $RUNNING_TIME" | tee -a $LOG_FILE
    let "TOTAL_SECONDS += $SECONDS"
}


mkdir -p log
LOG_FILE="log/run.log"
echo "===============================" >> $LOG_FILE
echo "date: $(date)" >> $LOG_FILE
echo "version: $VERSION" >> $LOG_FILE
TOTAL_SECONDS=0
# "CONFIG##*/" is the basename of CONFIG
while getopts "abc" flag; do
  case "$flag" in
    a)
      SECONDS=0
      TYPE="debug"

      CONFIG="dataset_model"
      EXP="dataset_model"
      RUN_ID=${VERSION}

      # use optuna to find the best h.p.
      python3 mains/main.py --optuna --mp -c "configs/$CONFIG.json" --mode train \
          --run_id $RUN_ID --name $EXP
      python3 mains/main.py -c "saved/$EXP/$RUN_ID/best_hp/${CONFIG##*/}.json" --mode test \
          --resume "saved/$EXP/$RUN_ID/best_hp/fold_0_model_best.pth" --run_id $RUN_ID --k_fold 1 --bootstrapping

      # given h.p. with k_fold = 1
      python3 mains/main.py -c "configs/$CONFIG.json" --mode train \
          --run_id $RUN_ID --name $EXP --k_fold 1
      python3 mains/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test \
          --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID --bootstrapping

      time_log
      ;;
    b)
      SECONDS=0
      TYPE="preprocess"

      python3 preprocess.py

      time_log
      ;;
    c)
      SECONDS=0
      TYPE="run"

      CONFIG="dataset_model"
      EXP="dataset_model"
      RUN_ID=${VERSION}

      time_log
      ;;
    *)
      usage
      ;;
  esac
done

TOTAL_TIME=$(($TOTAL_SECONDS/86400))" days, "$(date +%T -d "1/1 + $TOTAL_SECONDS sec")
echo -e "---------------------------------" | tee -a $LOG_FILE
echo -e "total running time: $TOTAL_TIME" | tee -a $LOG_FILE
