import logging
from pathlib import Path

from logger import setup_logging
from utils import ensure_dir, read_json, write_json


class ConfigParser:
    def __init__(self, args, resume=None):
        """
        Initialize this class from config.json. Used in train, test.
        """
        config_path = Path(args.config)
        self._config = read_json(config_path)
        self.resume = resume

        if args.mode == 'train':
            save_dir = Path(self.config['trainer']['save_dir'])
            exp_dir  = save_dir / self.config['name']
            self.save_dir = dict()
            for dir_name in ['fig', 'log', 'model']:
                dir_path = exp_dir / dir_name
                ensure_dir(dir_path)
                self.save_dir.update({dir_name: dir_path})

            # save config file to the experiment dirctory
            write_json(self.config, exp_dir / 'config.json')

            # configure logging module
            setup_logging(self.save_dir['log'])
            self.log_levels = {
                0: logging.WARNING,
                1: logging.INFO,
                2: logging.DEBUG
            }

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a class/function handle with the name given as 'type' in config, and returns the
        instance/function initialized with given arguments.
        `object/function = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object/function = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # read-only attributes
    @property
    def config(self):
        return self._config
