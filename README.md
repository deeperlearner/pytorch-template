# PyTorch Template Project
A pytorch template files generator, which supports multiple instances of dataset, dataloader, model, optimizer, loss, optimizer and lr_scheduler.

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

### module/type
When there are more than one module, for example,
- `data_loaders/first_loader.py` and `data_loaders/second_loader.py`
- `trainers/first_trainer.py` and `trainers/second_trainer.py`
- `models/model1.py` and `models/model2.py`

Each of them has some classes. In `parse_config.py`, ConfigParser.init_obj() can automatically import the specified class by using importlib.

### AUROC/AUPRC
In metric part, I add two commonly used metrics AUROC/AUPRC. These two metrics need to be computed on whole epoch, so the compute method is different from accuracy.

### MetricTracker
Continue from AUROC/AUPRC, I revise the MetricTracker, which is moved to `models/metric.py`.
The MetricTracker can record both accuracy-like metric (metrics_iter) and AUROC-like (metrics_epoch) metric.

### Cross validation
Cross validation are supported.
Class `Cross_Valid` in `base/base_dataloader.py` records the index of cross validation.
The models and metric results of each fold are saved.
`ensemble.py` can ensemble k-fold validation results.
Also, multi-process cross validation is supported, which allows you to run many folds simultaneously in the background.
The multi-processing is handled by `scripts/run/run.sh`. You can decide how many processes you want to run at a time by edit the shell script.

:warning: Caveat: If your dataset is large, running many processes may cost a lot of RAM.
Be careful to adjust the number of processes and the number of workers in the `data_loaders` part of config.

### Examples
I add some example codes to use the above features.
- MNIST dataset
- ImageNet dataset (The data need to be downloaded by yourself)
- Adult dataset

try `bash scripts/run/examples.sh run_all` to run all examples.

## Folder Structure
  ```
  Pytorch-Template/
  │
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── ensemble.py - ensemble k-fold results
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   └── base_trainer.py
  │
  ├── configs/ - configurations for training
  │   ├── dataset_model.json
  │   └── examples/
  │       └── *.json
  │
  ├── data/ - default directory for storing input data
  │
  ├── data_loaders/ - anything about data loading goes here
  │   ├── data_loader.py
  │   └── examples/
  │       └── *_loader.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │
  ├── models/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   ├── loss.py
  │   └── examples/
  │       └── *.py
  │
  ├── saved/ - train information
  │   └── EXP_name/
  │       └── run_id/
  │           ├── log/ - training log and logdir for tensorboard
  │           ├── metrics_best/ - training best metric results
  │           ├── models/ - trained models are saved here
  │           └── dataset_model.json - backup config file when start training
  │
  ├── output/ - test information
  │   └── EXP_name/
  │       └── run_id/
  │           ├── fig/ - figures
  │           ├── log/ - testing log
  │           └── metric/ - testing metric results
  │
  ├── scripts/ - scripts for *.sh
  │   ├── new_project/
  │   │   ├── copy_exclude - exclude file when create new project
  │   │   └── torch_new_project.sh - initialize new project with template files
  │   ├── run/
  │   │   ├── examples.sh - bash script for running examples
  │   │   └── run.sh - bash script for running experiment
  │   └── version_update/
  │       ├── file_list - files to preserve between update
  │       └── version_update.sh - transfer old version files to new version directory
  │
  ├── trainers/ - trainers
  │   ├── trainer.py
  │   └── examples/
  │       └── *_trainer.py
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── examples/
          └── *.py
  ```

## Count Lines of Codes
- wc
`wc -l **/*.* *.*`
- cloc
`sudo apt install cloc`
`cloc --vcs=git --by-file`

## Usage
There are some examples config files in `config/examples/`. Try `bash run_examples.sh` to run code.

### Config file format
Config files are in `.json` format, `dataset_model.json`:
```javascript
{
    "n_gpu": 1,
    "root_dir": "./",
    "name": "dataset_model",

    "datasets": {
        "train": {
            "data": {
                "module": ".data_loader",
                "type": "MyDataset",
                "kwargs": {
                    "data_dir": "./data",
                    "label_path": null,
                    "mode": "train"
                }
            }
        },
        "valid": {
        },
        "test": {
            "data": {
                "module": ".data_loader",
                "type": "MyDataset",
                "kwargs": {
                    "data_dir": "./data",
                    "label_path": null,
                    "mode": "test"
                }
            }
        }
    },
    "data_loaders": {
        "train": {
            "data": {
                "module": ".data_loader",
                "type": "BaseDataLoader",
                "kwargs": {
                    "validation_split": 0.2,
                    "DataLoader_kwargs": {
                        "batch_size": 64,
                        "shuffle": true,
                        "num_workers": 4
                    },
                    "do_transform": true
                }
            }
        },
        "valid": {
        },
        "test": {
            "data": {
                "module": ".data_loader",
                "type": "DataLoader",
                "kwargs": {
                    "batch_size": 64,
                    "shuffle": false,
                    "num_workers": 4
                },
                "do_transform": true
            }
        }
    },
    "models": {
        "model": {
            "module": ".model",
            "type": "MyModel"
        }
    },
    "losses": {
        "loss": {
            "is_ftn": true,
            "balanced": false,
            "type": "nll_loss"
        }
    },
    "metrics": {
        "per_iteration": ["accuracy"],
        "per_epoch": ["AUROC", "AUPRC"]
    },
    "optimizers": {
        "model": {
            "type": "Adam",
            "kwargs": {
                "lr": 0.001
            }
        }
    },
    "lr_schedulers": {
        "model": {
            "type": "StepLR",
            "kwargs": {
                "step_size": 50,
                "gamma": 0.1
            }
        }
    },
    "trainer": {
        "module": ".trainer",
        "type": "Trainer",
        "k_fold": 5,
        "fold_idx": 0,
        "kwargs": {
            "finetune": false,
            "epochs": 2,
            "len_epoch": null,

            "save_period": 5,
            "save_the_best": true,
            "verbosity": 2,

            "monitor": "max val_accuracy",
            "early_stop": 0,

            "tensorboard": false
        }
    }
}
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:
  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization

### Project initialization
Use the `torch_new_project.sh` script to make your new project directory with template files.

Add this line to ~/.bashrc:
`export Pytorch_Template=/path/to/Pytorch_Template`

Add symbolic link at /usr/local/bin so that you can run this script everywhere.

`sudo ln -s $Pytorch_Template/scripts/new_project/torch_new_project.sh /usr/local/bin/torch_new_project`

`torch_new_project ProjectName` produces a new project folder named 'ProjectName' will be made.
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
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
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

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

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
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": {
      "per_iteration": ["accuracy", "top_k_acc"],
      "per_epoch": ["AUROC", "AUPRC"]
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
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

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
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs

- [ ] Revise errors in trainer/examples and test\_examples/
- [ ] Support more tensorboard functions

## Acknowledgements
This project is forked and enhanced from the project [pytorch-template](https://github.com/victoresque/pytorch-template)
