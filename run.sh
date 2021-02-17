# This script run train and test
# bash run.sh

# train
python3 train.py -c config/dataset_model.json --run_id 0

# test
python3 test.py -c config/dataset_model.json --resume saved/dataset_model/0/model/model_best.pth
