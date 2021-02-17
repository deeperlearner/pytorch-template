# This script run train and test for examples experiment
# bash run_examples.sh

# MNIST_LeNet
#python3 train.py -c config/examples/MNIST_LeNet.json --run_id 0
#python3 test.py -c config/examples/MNIST_LeNet.json --resume saved/MNIST_LeNet/0/model/model_best.pth

# MNIST_cv_LeNet
#python3 train.py -c config/examples/MNIST_cv_LeNet.json --run_id 0
#python3 test.py -c config/examples/MNIST_cv_LeNet.json --resume saved/MNIST_cv_LeNet/0/model/model_best.pth

# ImageNet_VGG16
#python3 train.py -c config/examples/ImageNet_VGG16.json --run_id 0
#python3 test.py -c config/examples/ImageNet_VGG16.json --resume saved/ImageNet_VGG16/0/model/model_best.pth

# Adult_logistic
python3 train.py -c config/examples/Adult_logistic.json --run_id 0
python3 test.py -c config/examples/Adult_logistic.json --resume saved/Adult_logistic/0/model/model_best.pth
