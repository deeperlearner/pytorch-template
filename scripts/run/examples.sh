#!/bin/bash
# This script run train and test for examples
# bash ./scripts/run/examples.sh $1
# $1: debug or run_all

MODE=$1

if [ "$MODE" = "debug" ]; then
    CONFIG="examples/Adult_logistic"
    EXP="Adult_logistic"
    RUN_ID="debug"
    python3 mains/train/train.py -c "configs/$CONFIG.json" --run_id $RUN_ID
    python3 mains/test/test.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID
else
    # MNIST_LeNet
    CONFIG="examples/MNIST_LeNet"
    EXP="MNIST_LeNet"
    RUN_ID="0"
    python3 mains/train/train.py -c "configs/$CONFIG.json" --run_id $RUN_ID
    python3 mains/test/test.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID

    # ImageNet_VGG16 (need to download ImageNet dataset)
    CONFIG="examples/ImageNet_VGG16"
    EXP="ImageNet_VGG16"
    RUN_ID="0"
    python3 mains/train/train.py -c "configs/$CONFIG.json" --run_id $RUN_ID
    # no test data
    # python3 mains/test/test.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID

    # Adult_logistic cv by single-process
    CONFIG="examples/Adult_logistic"
    EXP="Adult_logistic"
    RUN_ID="0"
    python3 mains/train/train.py -c "configs/$CONFIG.json" --run_id $RUN_ID
    python3 mains/test/test.py -c "saved/$EXP/$RUN_ID/${CONFIG##*/}.json" --resume "saved/$EXP/$RUN_ID/model/model_best.pth" --run_id $RUN_ID

    # Not implemented yet
    # I'm going to try `import torch.multiprocessing as mp`
    ## Adult_logistic cv by multi-process
    #CONFIG="examples/Adult_logistic"
    #EXP="Adult_logistic"
    #RUN_ID="1"
fi
