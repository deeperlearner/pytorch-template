import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime

from logger import setup_logging
from utils import ensure_dir, read_json, write_json


class ConfigParser:
    def __init__(self, config_json: str, run_args: dict, modification=None):
        """
        class to parse configuration json file. Handles hyperparameters for training,
        initializations of modules, checkpoint saving and logging module.
        :param config_json: String, path to the config file.
        :param run_args: Dict, running arguments including resume, mode, run_id, log_name.
            - resume: String, path to the checkpoint being loaded.
            - mode: String, 'train', 'test' or 'inference'.
            - run_id: Unique Identifier for training processes. Used to save checkpoints and training log.
                     Timestamp is being used as default
            - log_name: Change info.log into <log_name>.log.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        """
        # load config file and apply modification
        config = read_json(config_json)
        self._config = _update_config(config, modification)
        self.resume = run_args['resume']
        self.mode = run_args['mode']
        run_id = run_args['run_id']
        log_name = run_args['log_name']

        if self.mode == 'train':
            save_dir = Path(self.config['save_dir'])
            if run_id is None: # use timestamp as default run-id
                run_id = datetime.now().strftime(r'%m%d_%H%M%S')
            exp_dir  = save_dir / self.config['name'] / run_id

            self.save_dir = dict()
            for dir_name in ['log', 'model']:
                dir_path = exp_dir / dir_name
                ensure_dir(dir_path)
                self.save_dir.update({dir_name: dir_path})

            # save config file to the experiment dirctory
            write_json(self.config, exp_dir / os.path.basename(config_json))

            # configure logging module
            setup_logging(self.save_dir['log'], log_config=self.config['log_config'], filename=log_name)
        elif self.mode == 'test':
            # configure logging module
            log_dir = Path('log')
            ensure_dir(log_dir)
            setup_logging(log_dir, log_config=self.config['log_config'], filename=log_name)

        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)

        args = args.parse_args()

        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg
        config_json = Path(args.config)

        run_args = dict()
        run_args['resume'] = Path(args.resume) if args.resume is not None else None
        run_args['mode'] = args.mode
        run_args['run_id'] = args.run_id
        run_args['log_name'] = args.log_name

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}

        return cls(config_json, run_args, modification)

    def init_obj(self, kind, name, module, *args, **kwargs):
        """
        Finds a class handle with the name given as 'type' in config, and returns the
        instance initialized with given arguments.
        `object = config.init_obj('kind', 'name', module, a, b=1)`
        is equivalent to
        `object = module.name(obj_args, a, b=1)`
        """
        try:
            module_name = self[kind][name]['module']
            class_name = self[kind][name]['type']
            obj = reduce(getattr, [module , module_name, class_name])
        except KeyError:
            class_name = self[kind][name]['type']
            obj = getattr(module, class_name)

        obj_args = self[kind][name]['args'] # type: dict

        return obj(obj_args, *args, **kwargs)

    def init_ftn(self, kind, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.
        `function = config.init_ftn('kind', 'name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        class_name = self[kind][name]['type']
        ftn = getattr(module, class_name)
        module_args = dict(self[kind][name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)

        return partial(ftn, *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                    self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # read-only attributes
    @property
    def config(self):
        return self._config

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for key, value in modification.items():
        if value is not None:
            _set_by_path(config, key, value)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
