from logger import get_logger
from mains.train import train
from parse_config import ConfigParser
from utils import msg_box, consuming_time


objective_results = []
def objective(trial):
    # TODO: hyperparameters search spaces
    optimizer = trial.suggest_categorical("optimizer", ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)

    modification = {"optimizers;model;type": optimizer,
                    "optimizers;model;kwargs;lr": lr}
    config = ConfigParser(modification)
    logger = get_logger('optuna')
    max_min, mnt_metric = config['trainer']['kwargs']['monitor'].split()

    result = train(config)
    best = result.at[mnt_metric, 'mean']
    objective_results.append(best)
    msg = msg_box("Optuna progress")
    i, N = len(objective_results), config.run_args.optuna_trial
    msg += f"\ntrial: ({i}/{N})"

    if (max_min == 'max' and best >= max(objective_results) or
            max_min == 'min' and best <= min(objective_results)):
        msg += "\nBackuping best hyperparameters config and model..."
        config.backup()
        config.cp_models()
    logger.info(msg)

    return best
