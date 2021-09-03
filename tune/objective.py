from logger import change_log_name, get_logger
from mains import train, train_mp
from parse_config import ConfigParser
from utils import msg_box, consuming_time

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
    change_log_name(config, log_name="optuna.log")
    logger = get_logger("optuna")
    max_min, mnt_metric = config["trainer"]["kwargs"]["monitor"].split()
    k_fold = config["cross_validation"]["k_fold"]
    msg = msg_box("Optuna progress")
    i, N = len(objective_results), config["optuna"]["n_trials"]
    msg += f"\nTrial: ({i}/{N-1})"
    logger.info(msg)

    if config.run_args.mp:
        # train with multiprocessing on k_fold
        results = train_mp(config)
        result = sum(results) / len(results)
    else:
        result = train(config)
    objective_results.append(result)

    if (
        max_min == "max"
        and result >= max(objective_results)
        or max_min == "min"
        and result <= min(objective_results)
    ):
        msg = "\nBackuping best hyperparameters config and model..."
        config.backup(best_hp=True)
        config.cp_models()
        logger.info(msg)

    return result
