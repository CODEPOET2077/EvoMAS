"""Global configuration object (mmengine ``Config`` loaded from ``configs/*.py``)."""

import os
from argparse import Namespace

from dotenv import load_dotenv
from mmengine import Config as MMConfig

from src.logger import logger
from src.utils import Singleton, assemble_project_path

load_dotenv(verbose=True)


def process_general(config: MMConfig) -> MMConfig:
    """Resolve output paths under ``<workdir>/<tag>/``."""
    config.exp_path = assemble_project_path(os.path.join(config.workdir, config.tag))
    os.makedirs(config.exp_path, exist_ok=True)
    config.log_path = os.path.join(config.exp_path, getattr(config, "log_path", "log.txt"))
    if "save_path" in config:
        config.save_path = os.path.join(config.exp_path, config.save_path)
    logger.info(f"| Experiment directory: {config.exp_path}")
    return config


class Config(MMConfig, metaclass=Singleton):
    def __init__(self):
        super().__init__()

    def init_config(self, config_path: str, args: Namespace) -> None:
        """Load ``config_path`` and merge ``--cfg-options`` overrides from the command line."""
        mmconfig = MMConfig.fromfile(filename=assemble_project_path(config_path))
        cfg_options = dict(getattr(args, "cfg_options", None) or {})
        for key, value in vars(args).items():
            if key not in ("config", "cfg_options") and value is not None:
                cfg_options[key] = value
        mmconfig.merge_from_dict(cfg_options)
        mmconfig = process_general(mmconfig)
        self.__dict__.update(mmconfig.__dict__)


config = Config()
