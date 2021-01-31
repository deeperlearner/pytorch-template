# This script run train and test
# bash run_examples.sh

# train
#python3 train.py -c config/examples/MNIST_LeNet.json --run_id 0
python3 train.py -c config/examples/MNIST_cv_LeNet.json --run_id 0
#python3 train.py -c config/examples/ImageNet_VGG16.json --run_id 0

# test
#python3 test.py -c config/examples/MNIST_LeNet.json --resume saved/MNIST_LeNet/0/model/model_best.pth
python3 test.py -c config/examples/MNIST_cv_LeNet.json --resume saved/MNIST_LeNet/0/model/model_best.pth
#python3 test.py -c config/examples/ImageNet_VGG16.json --resume saved/ImageNet_VGG16/0/model/model_best.pth

# inference
#python3 test.py -c config/config.json --resume model_best.pth
