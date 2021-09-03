#!/usr/bin/env bash
# ------------------
#  PyTorch Template
# ------------------
# Repository    : https://github.com/deeperlearner/pytorch-template
VERSION="v4.0.0"


# This script run train and test for examples
usage() { echo "Usage: $0 [-dpr]" 1>&2; exit 1; }

# record execution time to log
time_log() {
    RUNNING_TIME=$(date +%T -d "1/1 + $SECONDS sec")
    echo -e "---------------------------------" | tee -a $LOG_FILE
    echo -e "$TYPE running time: $RUNNING_TIME" | tee -a $LOG_FILE
    let "TOTAL_SECONDS += $SECONDS"
}


mkdir -p log
LOG_FILE="log/examples.log"
echo "===============================" >> $LOG_FILE
echo "version: $VERSION" >> $LOG_FILE
TOTAL_SECONDS=0
# "CONFIG##*/" is the basename of CONFIG
while getopts "dpr" flag; do
  case "$flag" in
    d)
      SECONDS=0
      TYPE="debug"

      CONFIG="examples/Adult_logistic"
      EXP="Adult_logistic"
      RUN_ID=${VERSION}
      # use optuna to find the best h.p.
      # python3 mains/main.py --optuna -c "configs/$CONFIG.json" --mode train --run_id $RUN_ID --name $EXP
      # python3 mains/main.py -c "saved/$EXP/$RUN_ID/best_hp/${CONFIG##*/}.json" --mode test \
      #     --resume "saved/$EXP/$RUN_ID/best_hp/model_best.pth" --run_id $RUN_ID

      # given h.p. with k_fold = 1
      # python3 mains/main.py -c "configs/$CONFIG.json" --mode train --run_id $RUN_ID
      # python3 mains/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test \
      #     --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID --bootstrapping

      CONFIG="examples/Adult_logistic"
      EXP="Adult_logistic_mp"
      RUN_ID=${VERSION}
      python3 mains/main.py --optuna --mp -c "configs/$CONFIG.json" --mode train --run_id $RUN_ID --name $EXP
      python3 mains/main.py -c "saved/$EXP/$RUN_ID/best_hp/${CONFIG##*/}.json" --mode test \
          --resume "saved/$EXP/$RUN_ID/best_hp/model_best.pth" --run_id $RUN_ID

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

      # MNIST_LeNet
      CONFIG="examples/MNIST_LeNet"
      EXP="MNIST_LeNet"
      RUN_ID=${VERSION}
      python3 mains/main.py -c "configs/$CONFIG.json" --mode train --run_id $RUN_ID
      python3 mains/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test \
          --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID

      # ImageNet_VGG16 (need to download ImageNet dataset)
      CONFIG="examples/ImageNet_VGG16"
      EXP="ImageNet_VGG16"
      RUN_ID=${VERSION}
      python3 mains/main.py -c "configs/$CONFIG.json" --mode train --run_id $RUN_ID
      # no test data
      # python3 mains/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test \
      #     --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID

      # Adult_logistic cv by single-process
      CONFIG="examples/Adult_logistic"
      EXP="Adult_logistic"
      RUN_ID=${VERSION}
      python3 mains/main.py -c "configs/$CONFIG.json" --mode train --run_id $RUN_ID
      python3 mains/main.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --mode test \
          --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID

      # Not implemented yet
      # I'm going to try `import torch.multiprocessing as mp`
      ## Adult_logistic cv by multi-process
      #CONFIG="examples/Adult_logistic"
      #EXP="Adult_logistic"
      #RUN_ID=${VERSION}

      time_log
      ;;
    *)
      usage
      ;;
  esac
done

TOTAL_TIME=$(date +%T -d "1/1 + $TOTAL_SECONDS sec")
echo -e "---------------------------------" | tee -a $LOG_FILE
echo -e "total running time: $TOTAL_TIME" | tee -a $LOG_FILE
