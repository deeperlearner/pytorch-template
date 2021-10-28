import os
import sys
import argparse
import collections
import time

import optuna

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from logger import get_logger
from parse_config import ConfigParser
from mains import train, train_mp, test
from utils import msg_box, consuming_time, get_by_path


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="training")

    # crutial args executed in scripts
    run_args = args.add_argument_group("run_args")
    run_args.add_argument("--optuna", action="store_true")
    run_args.add_argument("--mp", action="store_true", help="multiprocessing")
    run_args.add_argument("--n_jobs", default=3, type=int, help="number of jobs running at the same time")
    run_args.add_argument("-c", "--config", default="configs/config.json", type=str)
    run_args.add_argument("--mode", default="train", type=str)
    run_args.add_argument("--resume", default=None, type=str)
    run_args.add_argument("--run_id", default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    mod_args = args.add_argument_group("mod_args")
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--name"], type=str, target="name"),
        CustomArgs(["--k_fold"], type=int, target="cross_validation;k_fold"),
        CustomArgs(
            ["--num_workers"],
            type=int,
            target="data_loaders;train;data;kwargs;DataLoader_kwargs;num_workers",
        ),
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target="optimizers;model;args;lr"
        ),
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target="data_loaders;train;data;args;DataLoader_kwargs;batch_size",
        ),
    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, type=opt.type)

    # config.test_args: additional arguments for testing
    test_args = args.add_argument_group("test_args")
    test_args.add_argument("--bootstrapping", action="store_true")
    test_args.add_argument("--bootstrap_times", default=1000, type=int)
    test_args.add_argument("--output_path", default=None, type=str)

    config = ConfigParser.from_args(args, options)
    config.set_log()
    logger = get_logger("main")
    mode = config.run_args.mode
    msg = msg_box(mode.upper())
    logger.debug(msg)

    if mode == "train":
        if config.run_args.optuna:
            keys = ["trainers", "trainer", "kwargs", "monitor"]
            max_min, mnt_metric = get_by_path(config, keys).split()
            objective = config.init_obj(["optuna"])
            n_trials = config["optuna"]["n_trials"]

            config.set_log(log_name="optuna.log")
            logger = get_logger("optuna")
            optuna.logging.enable_propagation()
            optuna.logging.disable_default_handler()
            direction = "maximize" if max_min == "max" else "minimize"
            start = time.time()
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=n_trials)

            msg = msg_box("Optuna result")
            end = time.time()
            total_time = consuming_time(start, end)
            msg += f"\nConsuming time: {total_time}."
            msg += f"\nM{direction[1:-3]}al {mnt_metric}: {study.best_value:.6f}"
            msg += f"\nBest hyperparameters: {study.best_params}"
            logger.info(msg)
        else:
            if config.run_args.mp:
                train_mp(config)
            else:
                train(config)
    elif mode == "test":
        test(config)
