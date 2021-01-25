# This script run train and test
# bash run.sh

# train
python3 train.py -c config/examples/MNIST_LeNet.json --run_id 0
#python3 train.py -c config/examples/ImageNet_VGG16.json --run_id 0

# test
#python3 test.py -c config/config.json --resume saved/Mnist/0/model/model_best.pth --output_path output/Mnist/0

# inference
#python3 test.py -c config/config.json --resume model_best.pth
