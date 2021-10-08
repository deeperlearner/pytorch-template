# PyTorch Template for all kinds of ML Projects

A pytorch template files generator, which supports multiple instances of dataset, dataloader, model, loss, optimizer and lr_scheduler.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [PyTorch Template Project](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
		* [Dataset](#data-loader)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss](#loss)
		* [Metrics](#metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [Tensorboard Visualization](#tensorboard-visualization)
	* [Contribution](#contribution)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements

* Bash (Linux)
* Python >= 3.6
* requirements.txt
* [Apex](https://github.com/NVIDIA/apex#quick-start)

## Features

* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for convenient parameter tuning.
* Support multi-dataset, multi-dataloader, multi-model, multi-optimizer, multi-loss, multi-optimizer and multi-lr\_scheduler.
And all of above can be constructed in `.json` config!
* By adding symbolic to /usr/local/bin, you can execute `torch_new_project` under all paths.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and initialize all kinds of objects.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` : currently not implemented.
* Additional features compared with [pytorch-template](https://github.com/victoresque/pytorch-template):

### Enable multiple instances in datasets, data_loaders, models, losses, optimizers, lr_schedulers

Multiple datasets like domain adaption training will use source dataset and target dataset, so do data_loaders.
Multiple models like GAN. Generator and Discriminator.
Multiple losses, optimizers, lr_schedulers can be found in many ML papers.

### train/valid/test

If the paths of train/valid/test are already given, they can be directly put in the section in datasets, data_loaders.

This is the flow chart of this template:
![flow chart](https://i.imgur.com/TlG3Ayt.png)

### module/type

When there are more than one module, for example,
- `data_loaders/first_loader.py` and `data_loaders/second_loader.py`
- `trainers/first_trainer.py` and `trainers/second_trainer.py`
- `models/model1.py` and `models/model2.py`

Each of them has some classes. In `parse_config.py`, `ConfigParser.init_obj()` can automatically import the specified class by using `importlib`.

### AUROC/AUPRC

In metric part, I add two commonly used metrics AUROC/AUPRC. These two metrics need to be computed on whole epoch, so the compute method is different from accuracy.

### MetricTracker

Continue from AUROC/AUPRC, I revise the MetricTracker, which is moved to `models/metric.py`.
The MetricTracker can record both iteration-based metrics (iter_record) and epoch-based metrics (epoch_record).

### Cross validation

Cross validation is supported.
Class `Cross_Valid` in `mains/cross_validation.py` records the index of cross validation.
The model and metric results of each fold are saved.

Also, multi-process cross validation is supported, which allows you to run many folds simultaneously.
To enable multiprocessing, add flag `--mp`.
You can decide how many processes you want to run at a time by specifying `--n_jobs <int>`.

:warning: Caveat: If your dataset is large, running many processes may cost a lot of RAM.
Be careful to adjust the number of processes and the number of workers in the `data_loaders` part of config.

### [Apex](https://github.com/NVIDIA/apex)

RTX GPU can use apex to do automatic mixed precision training.

### [Optuna](https://github.com/optuna/optuna)

Use optuna to find best hyperparameters.
  ```
  python3 mains/main.py -c config.json --mode train --optuna
  ```

## Folder Structure

  ```
  Pytorch-Template/
  │
  ├── base/ - abstract base classes
  │
  ├── configs/ - configurations for training
  │
  ├── data/ - default directory for storing input data
  │
  ├── data_loaders/ - anything about data loading goes here
  │
  ├── log/ - directory for storing running logs
  │
  ├── logger/ - module for tensorboard visualization and logging
  │
  ├── mains/ - main, train and test
  │
  ├── models/ - models, losses, and metrics
  │
  ├── output/ - test information
  │
  ├── saved/ - train information
  │
  ├── scripts/ - scripts for *.sh
  │
  ├── trainers/ - trainers
  │
  ├── tune/ - objectives for optuna h.p. search
  │
  ├── utils/ - small utility functions
  │
  └── parse_config.py - class to handle config file and cli options
  ```

## Count Lines of Codes

- wc
  ```
  wc -l **/*.* *.*
  ```
- cloc
  ```
  sudo apt install cloc
  cloc --vcs=git --by-file
  ```

## Usage

### Config file format

Config file is in `.json` format, see [`configs/dataset_model.json`](https://github.com/deeperlearner/Pytorch-Template/blob/master/configs/dataset_model.json):

### Examples

There are some examples config files in `configs/examples/*.json`.
- MNIST dataset
- ImageNet dataset (The data need to be downloaded by yourself)
- Adult dataset

Try `./scripts/examples.sh -r` to run example configs.

### Using config files

Modify the configurations in `.json` config files, then run:
  ```
  python mains/main.py --config config.json --mode train
  ```

### Resuming from checkpoints

You can resume from a previously saved checkpoint by:
  ```
  python mains/main.py --config config.json --mode train --resume path/to/checkpoint
  ```

### Using Multiple GPU

You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python mains/main.py --device 2,3 -c config.json --mode train
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python mains/main.py -c config.py --mode train
  ```

## Customization

### Project initialization

Use the `torch_new_project.sh` script to make your new project directory with template files.

1. Add this line to ~/.bashrc:
  ```
  export Pytorch_Template=/path/to/Pytorch_Template
  ```
1. Add symbolic link at /usr/local/bin so that you can run this script everywhere.
  ```
  sudo ln -s $Pytorch_Template/scripts/new_project/torch_new_project.sh /usr/local/bin/torch_new_project
  ```
1. `torch_new_project ProjectName` produces a new project folder named 'ProjectName' will be made.
This script will filter out unneccessary files listed in `copy_exclude`.

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target="optimizer;args;lr"),
      CustomArgs(['--bs', '--batch_size'], type=int, target="data_loader;args;batch_size")
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `"optimizer;args;lr"` because `config['optimizer']['args']['lr']` points to the learning rate.
`python mains/main.py -c config.json --mode train --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.


### Data Loader

* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to `data_loaders/examples/MNIST_loader.py` for an MNIST data loading example.

### Trainer

* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainers/trainer.py`

* **Example**

  Please refer to `trainers/trainer.py` for MNIST training.

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

### Model

* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `models/examples/LeNet.py` for a LeNet example.

### Loss

Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

### Metrics

Metric functions are located in `models/metric.py`.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": {
      "per_iteration": ["accuracy", "top_k_acc"],
      "per_epoch": ["AUROC", "AUPRC"],
      "pick_threshold": {
          "is_ftn": true,
          "type": "Youden_J",
          "kwargs": {
              "beta": 1.0
          }
      }
  }
  ```

### Additional logging

If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing

You can test trained model by running `mains/test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data (TODO)

To split validation data from a data loader, call `BaseDataLoader._train_valid_split()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `_train_valid_split()` method will modify the original data loader
**Note**: `_train_valid_split()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints

You can specify the name of the training session in config files:
  ```json
  name: MNIST_LeNet,
  ```

The checkpoints will be saved in `saved/name/run_id/model/checkpoint_epoch_n`, with timestamp in mmdd\_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
      'epoch': epoch,
      'models': {key: value.state_dict() for key, value in self.models.items()},
      'optimizers': {key: value.state_dict() for key, value in self.optimizers.items()},
      'monitor_best': self.mnt_best,
  }
  ```

### Tensorboard Visualization

This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by `pip install tensorboard>=1.14.0`.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/EXP/run_id/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules. 

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps.

## Contribution

Feel free to contribute any kind of function or enhancement, here the codes using black formatter

Code should use the [black](https://github.com/psf/black) to format the codes before committing.

## TODOs

- [ ] Revise errors in trainer/examples and test\_examples/
- [ ] Support more tensorboard functions

## Acknowledgements

This project is forked and enhanced from the project [pytorch-template](https://github.com/victoresque/pytorch-template)
