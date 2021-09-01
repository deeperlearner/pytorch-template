from multiprocessing import Queue

import torch.multiprocessing as _mp

from logger import get_logger
from mains import train_mp
from parse_config import ConfigParser
from utils import msg_box, consuming_time

mp = _mp.get_context('spawn')
objective_results = []


def objective(trial):
    # TODO: hyperparameters search spaces
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)

    modification = {
        "optimizers;model;type": optimizer,
        "optimizers;model;kwargs;lr": lr,
    }
    config = ConfigParser(modification)
    logger = get_logger("optuna")
    max_min, mnt_metric = config["trainer"]["kwargs"]["monitor"].split()
    k_fold = config["cross_validation"]["k_fold"]

    queue = Queue()
    processes = []
    results = []
    # multiprocessing on each fold_idx
    for fold_idx in range(k_fold):
        queue.put({"result": None})
        p = mp.Process(target=train_mp, args=(config, fold_idx, queue))
        p.start()
        processes.append(p)
    for p in processes:
        ret = queue.get()  # will block
        result = ret.at[mnt_metric, "mean"]
        results.append(result)
    for p in processes:
        p.join()

    avg_result = sum(results) / len(results)
    objective_results.append(avg_result)
    msg = msg_box("Optuna progress")
    i, N = len(objective_results), config["optuna"]["n_trials"]
    msg += f"\ntrial: ({i}/{N})"

    if (
        max_min == "max"
        and best >= max(objective_results)
        or max_min == "min"
        and best <= min(objective_results)
    ):
        msg += "\nBackuping best hyperparameters config and model..."
        config.backup(best_hp=True)
        config.cp_models()
    logger.info(msg)

    return best
