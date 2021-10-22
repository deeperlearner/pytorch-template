import os
import argparse
import logging
from pathlib import Path
from functools import partial
from datetime import datetime
import importlib
import glob
import shutil

from logger import setup_logging
from utils import ensure_dir, read_json, write_json, set_by_path, get_by_path


class ConfigParser:
    def __init__(self, modification=None):
        """
        class to parse configuration json file. Handles hyperparameters for training,
        initializations of modules, checkpoint saving and logging module.
        :param modification: Dict {keychain: value}, specifying position values to be replaced from config dict.
        """
        # run_args: self.run_args
        self.config_json = self.run_args.config
        config = read_json(Path(self.config_json))
        self.resume = (
            Path(self.run_args.resume) if self.run_args.resume is not None else None
        )
        run_id = self.run_args.run_id

        # mod_args: self.mod_args
        if modification is None:
            modification = {}
        modification.update(self.mod_args)
        if self.run_args.mp:  # lower trainer verbosity when multiprocessing
            modification.update({"trainers;trainer;kwargs;verbosity": 0})
        self._config = _update_config(config, modification)

        # test_args: self.test_args

        self.root_dir = self.config["root_dir"]
        save_name = {"train": "saved/", "test": "output/"}
        save_dir = Path(self.root_dir) / save_name[self.run_args.mode]
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        self.exp_dir = save_dir / self.config["name"] / run_id

        dirs = {"train": ["log", "model", "best_hp"], "test": ["log"]}
        self.save_dir = dict()
        for dir_name in dirs[self.run_args.mode]:
            dir_path = self.exp_dir / dir_name
            ensure_dir(dir_path)
            self.save_dir[dir_name] = dir_path

        if self.run_args.mode == "train":
            self.backup()

    @classmethod
    def from_args(cls, parser, options=""):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        args = parser.parse_args()

        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg

        modification = None
        for group in parser._action_groups:
            if group.title == "mod_args":
                # parse custom cli options into dictionary
                modification = {
                    opt.target: getattr(args, _get_opt_name(opt.flags))
                    for opt in options
                }
                cls.mod_args = modification
            else:
                group_dict = {
                    g.dest: getattr(args, g.dest, None) for g in group._group_actions
                }
                arg_group = argparse.Namespace(**group_dict)
                if group.title == "run_args":
                    cls.run_args = arg_group
                elif group.title == "test_args":
                    cls.test_args = arg_group

        return cls()

    @staticmethod
    def _update_kwargs(_config, kwargs):
        try:
            _kwargs = dict(_config["kwargs"])
        except KeyError:  # In case no arguments are specified
            _kwargs = dict()
        assert all(
            [k not in _kwargs for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        _kwargs.update(kwargs)
        return _kwargs

    def init_obj(self, keys, *args, **kwargs):
        """
        Returns an object or a function, which is specified in config[keys[0]]...[keys[-1]].
        In config[keys[0]]...[keys[-1]],
            'is_ftn': If True, return a function. If False, return an object.
            'module': The module of each instance.
            'type': Class name.
            'kwargs': Keyword arguments for the class initialization.
        keys is the list of config entries.
        Additional *args and **kwargs would be forwarded to obj()
        Usage: `objects = config.init_obj(['A', 'B', 'C'], a, b=1)`
        """
        obj_config = get_by_path(self, keys)
        module_name = obj_config["module"]
        module_obj = importlib.import_module(module_name)
        class_name = obj_config["type"]
        obj = getattr(module_obj, class_name)
        kwargs_obj = self._update_kwargs(obj_config, kwargs)

        if obj_config.get("is_ftn", False):
            return partial(obj, *args, **kwargs_obj)
        return obj(*args, **kwargs_obj)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    # configure logging module
    def set_log(self, log_name=None):
        setup_logging(
            self.save_dir["log"],
            root_dir=self.root_dir,
            filename=log_name,
        )

    # read-only attributes
    @property
    def config(self):
        return self._config

    # backup config file to the experiment dirctory
    def backup(self, best_hp=False):
        if best_hp:
            dir_path = self.exp_dir / "best_hp"
        else:
            dir_path = self.exp_dir
        write_json(self.config, dir_path / os.path.basename(self.config_json))

    # backup best models into best_hp/
    def cp_models(self):
        model_files = os.path.join(self.save_dir["model"], "*model_best.pth")
        for model_file in glob.glob(model_files):
            shutil.copy(model_file, self.save_dir["best_hp"])


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for key, value in modification.items():
        if value is not None:
            set_by_path(config, key, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")
