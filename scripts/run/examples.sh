#!/bin/bash
# This script run train and test for examples
# bash ./scripts/run/examples.sh $1
# $1: debug or run_all

MODE=$1
# cross validation by single-process
# $1: CONFIG
# $2: EXP
# $3: RUN_ID
cv_single () {
    SECONDS=0
    python3 train.py -c "configs/$1.json" --run_id $3
    python3 ensemble.py --k_fold 3 --metric_dir "saved/$2/$3/metrics_best" --log_dir "saved/$2/$3/log"
    time=$(date +%T -d "1/1 + $SECONDS sec")
    echo -e "============================\nTotal running time: $time" | tee -a "saved/$2/$3/log/info.log"
    python3 test.py -c "configs/$1.json" --resume "saved/$2/$3/model/model_best.pth" --run_id $3
    python3 ensemble.py --k_fold 3 --metric_dir "output/$2/$3/metric" --log_dir "output/$2/$3/log"
}

# cross validation by multi-process
# $1: CONFIG
# $2: EXP
# $3: RUN_ID
cv_multi () {
    SECONDS=0
    python3 train.py -c "configs/$1.json" --run_id $3 --log_name fold_1.log --fold_idx 1 &
    python3 train.py -c "configs/$1.json" --run_id $3 --log_name fold_2.log --fold_idx 2 &
    python3 train.py -c "configs/$1.json" --run_id $3 --log_name fold_3.log --fold_idx 3 &
    wait
    python3 ensemble.py --k_fold 3 --metric_dir "saved/$2/$3/metrics_best" --log_dir "saved/$2/$3/log"
    time=$(date +%T -d "1/1 + $SECONDS sec")
    echo -e "============================\nTotal running time: $time" | tee -a "saved/$2/$3/log/info.log"
    echo "train done!"
    python3 test.py -c "configs/$1.json" --resume "saved/$2/$3/model/model_best.pth" --run_id $3
    python3 ensemble.py --k_fold 3 --metric_dir "output/$2/$3/metric" --log_dir "output/$2/$3/log"
}

if [ "$MODE" = "debug" ]; then
    # Adult_logistic for debugging
    CONFIG="examples/Adult_logistic"
    EXP="Adult_logistic"
    RUN_ID="debug"
    # cv_single $CONFIG $EXP $RUN_ID
    python3 train.py -c "configs/$CONFIG.json" --run_id $RUN_ID
    python3 test.py -c "configs/$CONFIG.json" --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID
else  # "$MODE" = "run_all"
    # MNIST_LeNet
    CONFIG="examples/MNIST_LeNet"
    EXP="MNIST_LeNet"
    RUN_ID="0"
    python3 train.py -c "configs/$CONFIG.json" --run_id $RUN_ID
    python3 test.py -c "configs/$CONFIG.json" --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID

    # MNIST_LeNet (cv)
    CONFIG="examples/MNIST_cv_LeNet"
    EXP="MNIST_cv_LeNet"
    RUN_ID="0"
    cv_single $CONFIG $EXP $RUN_ID

    # ImageNet_VGG16 (too long don't run :/ ; you need to download ImageNet dataset)
    CONFIG="examples/ImageNet_VGG16"
    EXP="ImageNet_VGG16"
    RUN_ID="0"
    #python3 train.py -c "configs/$CONFIG.json" --run_id $RUN_ID
    #python3 test.py -c "configs/$CONFIG.json" --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID

    # Adult_logistic (cv)
    CONFIG="examples/Adult_logistic"
    EXP="Adult_logistic"
    RUN_ID="0"
    cv_single $CONFIG $EXP $RUN_ID

    # Adult_logistic (cv) use multiprocessing
    CONFIG="examples/Adult_logistic"
    EXP="Adult_logistic"
    RUN_ID="1"
    cv_multi $CONFIG $EXP $RUN_ID
fi
