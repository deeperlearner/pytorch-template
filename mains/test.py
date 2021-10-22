import os
import time

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from logger import get_logger
from mains import Cross_Valid, bootstrapping
import models.metric as module_metric
from models.metric import MetricTracker
from utils import prepare_device, get_by_path, msg_box, consuming_time

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

logger = get_logger("test")


def test(config):
    # setup GPU device if available, move model into configured device
    device, device_ids = prepare_device(config["n_gpu"])

    # datasets
    test_datasets = dict()
    keys = ["datasets", "test"]
    name = "data"
    test_datasets[name] = config.init_obj([*keys, name])

    repeat_time = config["cross_validation"]["repeat_time"]
    k_fold = config["cross_validation"]["k_fold"]

    results = pd.DataFrame()
    Cross_Valid.create_CV(repeat_time, k_fold)
    start = time.time()
    for k in range(k_fold):
        # data_loaders
        test_data_loaders = dict()
        keys = ["data_loaders", "test"]
        name = "data"
        dataset = test_datasets[name]
        loaders = config.init_obj([*keys, name], dataset)
        test_data_loaders[name] = loaders.test_loader

        # models
        if k_fold > 1:
            fold_prefix = f"fold_{k}_"
            dirname = os.path.dirname(config.resume)
            basename = os.path.basename(config.resume)
            resume = os.path.join(dirname, fold_prefix + basename)
        else:
            resume = config.resume
        logger.info(f"Loading model: {resume} ...")
        checkpoint = torch.load(resume)

        logger_model = get_logger("model", verbosity=0)
        models = dict()
        name = "model"
        model = config.init_obj(["models", name])
        logger_model.info(model)
        state_dict = checkpoint["models"][name]
        if config["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.to(device)
        models[name] = model

        # losses
        loss_fn = config.init_obj(["losses", "loss"])

        # metrics
        metrics_epoch = [
            getattr(module_metric, met) for met in config["metrics"]["per_epoch"]
        ]
        keys_epoch = [m.__name__ for m in metrics_epoch]
        test_metrics = MetricTracker([], keys_epoch)
        if "pick_threshold" in config["metrics"]:
            threshold = checkpoint["threshold"]
            setattr(module_metric, "THRESHOLD", threshold)
            logger.info(f"threshold: {threshold}")

        tester = config.init_obj(
            ["tester"], test_data_loaders, models, device, metrics_epoch, test_metrics
        )
        targets, outputs, test_log = tester.test()
        test_log = test_log["mean"].rename(k)
        results = pd.concat((results, test_log), axis=1)
        logger.info(test_log)

        if k_fold > 1:
            Cross_Valid.next_fold()

    msg = msg_box("result")

    end = time.time()
    total_time = consuming_time(start, end)
    msg += f"\nConsuming time: {total_time}."

    result = pd.DataFrame()
    result["mean"] = results.mean(axis=1)
    result["std"] = results.std(axis=1)
    msg += f"\n{result}"

    logger.info(msg)

    # bootstrap
    if config.test_args.bootstrapping:
        assert k_fold == 1, "k-fold ensemble and bootstrap are mutually exclusive."
        N = config.test_args.bootstrap_times
        bootstrapping(targets, outputs, metrics_epoch, test_metrics, repeat=N)
