import os
import time

import torch
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from logger import get_logger
from mains import Cross_Valid, Multiprocessor
import models.metric as module_metric
from utils import (
    prepare_device,
    get_by_path,
    msg_box,
    consuming_time,
    is_apex_available,
)

if is_apex_available():
    from apex import amp


def train_mp(config):
    k_fold = config["cross_validation"]["k_fold"]
    do_mp = config.run_args.mp
    n_jobs = config.run_args.n_jobs
    assert n_jobs <= k_fold, "n_jobs can not be more than k_fold."

    results = []
    fold_idx = 0
    while fold_idx < k_fold:
        mp = Multiprocessor()
        job_idx = 0
        while job_idx < n_jobs and fold_idx < k_fold:
            mp.run(train, config, do_mp, fold_idx)
            job_idx += 1
            fold_idx += 1
        ret = mp.wait()  # get results of processes
        results.extend(ret)

    return results


def train(config, do_mp=False, fold_idx=0):
    # different logging when multiprocessing
    if do_mp:
        config.set_log(log_name=f"fold_{fold_idx}.log")
    else:
        config.set_log()
    logger = get_logger("train")

    # setup GPU device if available, move model into configured device
    device, device_ids = prepare_device(config["n_gpu"])

    # datasets
    train_datasets = dict()
    valid_datasets = dict()
    ## train
    keys = ["datasets", "train"]
    for name in get_by_path(config, keys):
        train_datasets[name] = config.init_obj([*keys, name])
    ## valid
    valid_exist = False
    keys = ["datasets", "valid"]
    for name in get_by_path(config, keys):
        valid_exist = True
        valid_datasets[name] = config.init_obj([*keys, name])
    ## compute inverse class frequency as class weight
    if config["datasets"].get("imbalanced", False):
        target = train_datasets["data"].y_train  # TODO
        class_weight = compute_class_weight(
            class_weight="balanced", classes=target.unique(), y=target
        )
        class_weight = torch.FloatTensor(class_weight).to(device)
    else:
        class_weight = None

    # losses
    losses = dict()
    for name in config["losses"]:
        kwargs = {}
        if "balanced" in get_by_path(config, ["losses", name, "type"]):
            kwargs.update(class_weight=class_weight)
        losses[name] = config.init_obj(["losses", name], **kwargs)

    # metrics
    metrics_iter = [
        getattr(module_metric, met) for met in config["metrics"]["per_iteration"]
    ]
    metrics_epoch = [
        getattr(module_metric, met) for met in config["metrics"]["per_epoch"]
    ]
    if "pick_threshold" in config["metrics"]:
        metrics_threshold = config.init_obj(["metrics", "pick_threshold"])
    else:
        metrics_threshold = None

    torch_objs = {
        "datasets": {"train": train_datasets, "valid": valid_datasets},
        "losses": losses,
        "metrics": {
            "iter": metrics_iter,
            "epoch": metrics_epoch,
            "threshold": metrics_threshold,
        },
    }

    repeat_time = config["cross_validation"]["repeat_time"]
    k_fold = config["cross_validation"]["k_fold"]

    results = pd.DataFrame()
    Cross_Valid.create_CV(repeat_time, k_fold, fold_idx=fold_idx)
    start = time.time()
    for t in range(repeat_time):
        if k_fold > 1:  # cross validation enabled
            train_datasets["data"].split_cv_indexes(k_fold)
        # 1 loop for multi-process; k_fold loops for single-process
        k_time = 1 if do_mp else k_fold
        for k in range(k_time):
            # data_loaders
            train_data_loaders = dict()
            valid_data_loaders = dict()
            ## train
            keys = ["data_loaders", "train"]
            for name in get_by_path(config, keys):
                kwargs = {}
                if "imbalanced" in get_by_path(config, [*keys, name, "module"]):
                    kwargs.update(
                        class_weight=class_weight.cpu().detach().numpy(), target=target
                    )
                dataset = train_datasets[name]
                loaders = config.init_obj([*keys, name], dataset, **kwargs)
                train_data_loaders[name] = loaders.train_loader
                if not valid_exist:
                    valid_data_loaders[name] = loaders.valid_loader
            ## valid
            keys = ["data_loaders", "valid"]
            for name in get_by_path(config, keys):
                dataset = valid_datasets[name]
                loaders = config.init_obj([*keys, name], dataset)
                valid_data_loaders[name] = loaders.valid_loader

            # models
            models = dict()
            logger_model = get_logger("model", verbosity=1)
            for name in config["models"]:
                model = config.init_obj(["models", name])
                logger_model.info(model)
                model = model.to(device)
                if len(device_ids) > 1:
                    model = torch.nn.DataParallel(model, device_ids=device_ids)
                models[name] = model

            # optimizers
            optimizers = dict()
            for name in config["optimizers"]:
                trainable_params = filter(
                    lambda p: p.requires_grad, models[name].parameters()
                )
                optimizers[name] = config.init_obj(["optimizers", name], trainable_params)

            # learning rate schedulers
            lr_schedulers = dict()
            for name in config["lr_schedulers"]:
                lr_schedulers[name] = config.init_obj(
                    ["lr_schedulers", name], optimizers[name]
                )

            torch_objs.update(
                {
                    "data_loaders": {
                        "train": train_data_loaders,
                        "valid": valid_data_loaders,
                    },
                    "models": models,
                    "optimizers": optimizers,
                    "lr_schedulers": lr_schedulers,
                    "amp": None,
                }
            )

            # amp
            keys = ["trainers", "trainer", "kwargs", "apex"]
            if get_by_path(config, keys):
                # TODO: revise here if multiple models and optimizers
                models["model"], optimizers["model"] = amp.initialize(
                    models["model"], optimizers["model"], opt_level="O1"
                )
                torch_objs["amp"] = amp

            trainer = config.init_obj(
                ["trainers", "trainer"], torch_objs, config.save_dir, config.resume, device
            )
            train_log = trainer.train()
            results = pd.concat((results, train_log), axis=1)

            if k_time > 1:
                Cross_Valid.next_fold()

        if repeat_time > 1:
            Cross_Valid.next_time()

    msg = msg_box("result")

    end = time.time()
    total_time = consuming_time(start, end)
    msg += f"\nConsuming time: {total_time}."

    result = pd.DataFrame()
    result["mean"] = results.mean(axis=1)
    result["std"] = results.std(axis=1)
    msg += f"\n{result}"

    logger.info(msg)

    mnt_metric = trainer.mnt_metric
    result = result.at[mnt_metric, "mean"]

    return result
