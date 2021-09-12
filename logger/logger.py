import logging
import logging.config
from pathlib import Path

from utils import read_json


def setup_logging(
    save_dir,
    root_dir="./",
    filename=None,
    log_config="logger/logger_config.json",
    default_level=logging.INFO,
):
    """
    setup logging configuration
    """
    log_config = Path(root_dir) / log_config
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for handler_k, handler_v in config["handlers"].items():
            if "filename" in handler_v:
                if filename is None or handler_k != "info_file_handler":
                    handler_v["filename"] = str(save_dir / handler_v["filename"])
                else:
                    handler_v["filename"] = str(save_dir / filename)

        logging.config.dictConfig(config)
    else:
        print(
            "warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)


LOG_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def get_logger(name, verbosity=2):
    assert (
        verbosity in LOG_LEVELS
    ), "verbosity option {verbosity} is invalid. \
         Valid options are {LOG_LEVELS.keys()}."
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS[verbosity])
    return logger
