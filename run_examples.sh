# This script run train and test for examples
# bash run_examples.sh

# MNIST_LeNet
#python3 train.py -c config/examples/MNIST_LeNet.json --run_id 0
#python3 test.py -c config/examples/MNIST_LeNet.json --resume saved/MNIST_LeNet/0/model/model_best.pth

# MNIST_LeNet (cv)
#python3 train.py -c config/examples/MNIST_cv_LeNet.json --run_id 0
#python3 test.py -c config/examples/MNIST_cv_LeNet.json --resume saved/MNIST_cv_LeNet/0/model/model_best.pth
#python3 ensemble.py

# ImageNet_VGG16
#python3 train.py -c config/examples/ImageNet_VGG16.json --run_id 0
#python3 test.py -c config/examples/ImageNet_VGG16.json --resume saved/ImageNet_VGG16/0/model/model_best.pth

# Adult_logistic (cv)
#python3 train.py -c config/examples/Adult_logistic.json --run_id 0
#python3 test.py -c config/examples/Adult_logistic.json --resume saved/Adult_logistic/0/model/model_best.pth
#python3 ensemble.py

# Adult_logistic multithreading of cross validation
python3 train.py -c config/examples/Adult_logistic.json --run_id 0 --log_name fold_1.log --fold_idx 1 &
python3 train.py -c config/examples/Adult_logistic.json --run_id 0 --log_name fold_2.log --fold_idx 2 &
python3 train.py -c config/examples/Adult_logistic.json --run_id 0 --log_name fold_3.log --fold_idx 3 &
wait
#python3 ensemble.py
echo "all done!"
