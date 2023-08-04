from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from hydra import compose
from hydra.initialize import initialize_config_dir
from omegaconf import DictConfig

if TYPE_CHECKING:
    from logging import Logger


class LOG_LEVELS(IntEnum):
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    OFF = 0


parser = argparse.ArgumentParser()
parser.add_argument("exp_name", type=str, help="Name of experiment.")
parser.add_argument("cfg_name", type=str, help="Name of config file. Used as `$PWD/<--cfg_dir>/<cfg_name>.yaml`.")
parser.add_argument("--cfg_dir", default="configs", type=str)
parser.add_argument("--cfg_help", action="store_true", help="Show config and exit.")
parser.add_argument("--test", action="store_true", dest="test", default=False)
parser.add_argument("--local_rank", type=int, default=0, help="GPU local rank.")
parser.add_argument(
    "-l",
    "--log",
    type=str,
    default=LOG_LEVELS.WARN.name.lower(),
    choices=[e.name.lower() for e in list(LOG_LEVELS)],
    help="Set logging level.",
)
parser.add_argument("--log_file", type=str, default="", help="If set, logging will be output to file.")
parser.add_argument("--log_overwrite", action="store_true", help="Will overwrite log file from the beginning.")
parser.add_argument("-v", "--verbose", action="store_true", help="Shorthand for `--debug=info`.")
parser.add_argument("-vv", "--veryverbose", action="store_true", help="Shorthand for `--debug=debug`.")


def get_defaults():
    return parser.parse_known_args(sys.argv[1:])


def default_args():
    return get_defaults()[0]


def default_unknown():
    return get_defaults()[1]


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class BaseConfig:
    _frozen_properties = {}

    @staticmethod
    def mkdir(dir_path: Path):
        from .data_utils import valid_dir

        return valid_dir(dir_path)

    def merge_from(self, o: BaseConfig | DictConfig):
        attributes = set(dir(self))
        for key in dir(o):
            if key not in attributes or key in self._frozen_properties or key.upper() != key:
                continue
            s_attr = getattr(self, key)
            o_attr = getattr(o, key)
            if isinstance(s_attr, BaseConfig):
                s_attr.merge_from(o_attr)
            else:
                setattr(self, key, o_attr)
                if getattr(self, "cfg_help", default_args().cfg_help):
                    print(f"Update {key}: {s_attr} -> {o_attr}")

    def print_config(self, indent=0):
        for key in asdict(self).keys():
            if key.startswith("_"):
                continue
            elif key.endswith("DIR_BASE"):
                continue
            value = getattr(self, key)
            if isinstance(value, BaseConfig):
                print("  " * indent + f"{key}: [yellow]{value.__class__.__name__}[/]")
                value.print_config(indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class SchedulerConfig(BaseConfig):
    """
    # Scheduler Config
    """

    TYPE: str = "single_step"
    MILESTONES: list[int] = field(default_factory=lambda: [80, 120, 200, 240])
    STEP_SIZE: int = 100
    GAMMA: float = 0.5
    DECAY_EPOCHS: float = 0.5


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class DatasetConfig(BaseConfig):
    """
    # Dataset Base Class
    """

    DATASET: str = "CocoTrain"
    EPOCH: int = 10000
    NUM_WORKERS: int = 8
    SHUFFLE: bool = True

    SCHEDULER: SchedulerConfig = field(default_factory=SchedulerConfig)
    BATCH_SAMPLER: str = "default"
    BATCH_SIZE: int = 4


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class TrainConfig(DatasetConfig):
    """
    # Train Config
    """

    # Optimizer
    OPTIM: str = "adamw"
    LR: float = 1e-4
    WEIGHT_DECAY: float = 0.0

    SCHEDULER: SchedulerConfig = field(default_factory=SchedulerConfig)
    BATCH_SAMPLER: str = "default"
    BATCH_SIZE: int = 4


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class TestConfig(DatasetConfig):
    """
    # Test Config
    """

    EPOCH: int = 10000
    NUM_WORKERS: int = 1
    SHUFFLE: bool = False

    BATCH_SIZE: int = 4
    BATCH_SAMPLER: str = "default"


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class ModuleConfig(BaseConfig):
    """
    # Module Config
    """

    NAME: str = "resnet101"
    USE_LAYER: int = 4

    NUM_OUTPUT: int = 200


@dataclass(repr=False, eq=False, order=False, unsafe_hash=False, slots=True)
class Config(BaseConfig):
    """
    # Basic Config
    """

    EXP_NAME: str

    TRAIN: TrainConfig = field(default_factory=TrainConfig)
    TEST: TestConfig = field(default_factory=TestConfig)
    IS_TEST: bool = False

    def train_or_test(self, is_test: Optional[bool] = None):
        if is_test is None:
            is_test = self.IS_TEST
        return self.TEST if is_test else self.TRAIN

    # Inherit from other configs
    PARENT_CFG_NAME: str = "default"

    # Multiple GPUs
    DISTRIBUTED: bool = False
    LOCAL_RANK: int = 0
    FORCE_CPU: bool = False
    GPUS: list[int] = field(default_factory=lambda: list(range(8)))

    # Module
    MODULE: ModuleConfig = field(default_factory=ModuleConfig)

    # Training
    RESUME: bool = True
    LOAD_DATA_ON_MEMORY: bool = False

    # Logging
    NUM_ITER_PER_EP: int = -1
    SAVE_EP: int = 10
    SAVE_LATEST_EP: int = 5
    EVAL_EP: int = 20

    LOG_INTERVAL: int = 20
    RECORD_INTERVAL: int = 20

    # Others
    WHITE_BKGD: bool = False

    # Records
    TRAINED_MODEL_DIR_BASE: str = "data/trained_model"
    RECORD_DIR_BASE: str = "data/record"
    RESULT_DIR_BASE: str = "data/record"

    _frozen_properties = {"TRAINED_MODEL_DIR", "RECORD_DIR", "RESULT_DIR"}

    @property
    def TRAINED_MODEL_DIR(self):
        return self.mkdir(Path(self.TRAINED_MODEL_DIR_BASE) / self.EXP_NAME)

    @property
    def RECORD_DIR(self):
        return self.mkdir(Path(self.RECORD_DIR_BASE) / self.EXP_NAME)

    @property
    def RESULT_DIR(self):
        return self.mkdir(Path(self.RESULT_DIR_BASE) / self.EXP_NAME)


class Manager(Config, Logger):
    """
    # Manager

    The big granddaddy of all config options ad logging system.

    ## 1. Access Config Keys
    - Access the configuration keys defined in subclasses with `hydra` format.

    ```python
    print(mgr.EXP_NAME)
    print(mgr.TRAIN.DATASET)
    ```

    ## 2. Logging
    - Provides useful methods to log the output.

    ```python
    mgr.debug(f"Number of training data found: {len(datas)}")
    mgr.info("[green]Start training[/]")
    mgr.warn("This may take some time...")
    mgr.error("[red]Data NOT FOUND!![/]")
    ```
    """

    _logger: Optional[Logger] = None
    log_level: LOG_LEVELS = LOG_LEVELS.INFO
    veryverbose: bool = False

    _show_and_exit: bool = False
    _use_rich: bool = True
    _log_file: Optional[Path] = None

    def argparse_manager(self, args: argparse.Namespace):
        if (args.log or "").upper() in dir(LOG_LEVELS):
            self.log_level = LOG_LEVELS[args.log.upper()]
        if args.log_file:
            self._log_file = Path(args.log_file).expanduser().absolute()
            self.mkdir(self._log_file.parent)
            if args.log_overwrite:
                with self._log_file.open("w") as f:
                    f.write("")
        if args.verbose:
            self.log_level = LOG_LEVELS.DEBUG
        if args.veryverbose:
            self.veryverbose = True
            self.log_level = LOG_LEVELS.DEBUG
        self._show_and_exit = bool(args.cfg_help)
        return self

    @staticmethod
    def synchronize(log: Logger):
        """
        Helper function to synchronize (barrier) among all processes
        when using distributed training.
        """
        import torch.distributed as dist

        if not dist.is_available():
            log.warn(f"Distributed train failed: {dist.is_available()=}")
            return
        if not dist.is_initialized():
            log.warn(f"Distributed train failed: {dist.is_initialized()=}")
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            log.warn(f"Distributed train failed: {world_size=}")
            return
        log.info(f"{world_size=}")
        dist.barrier()

    def torch_setup_on_distributed(self, args: argparse.Namespace, log_base=Path("./logs")):
        import torch
        import torch.distributed as dist

        self.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count()
        torch.cuda.set_device(self.LOCAL_RANK)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        self.synchronize(self.log)
        if self.LOCAL_RANK != 0:
            from mltemplate.utils.data_utils import valid_dir

            args.log_file = valid_dir(log_base) / f"distributed-{self.LOCAL_RANK}.log"
            args.log_overwrite = True
            self._logger = None
            self.argparse_manager(args)
        self.log.info(f"Using DISTRIBUTED. {os.environ['LOCAL_RANK']=}, {os.environ['RANK']=}, {self.LOCAL_RANK=}")

    @staticmethod
    def setup_logger(level: int | LOG_LEVELS, use_rich=True, log_file: Optional[Path] = None):
        log = getLogger(__name__)
        log.setLevel(level)
        log.handlers.clear()
        if use_rich:
            from rich.highlighter import ReprHighlighter

            from .rich_logger import MyRichHandler

            ch = MyRichHandler(
                log_time_format="[%X]",
                markup=True,
                highlighter=ReprHighlighter(),
            )
            ch.setLevel(level)
            log.addHandler(ch)
        else:
            from logging import FileHandler, Formatter

            fh = FileHandler(str(log_file))
            fh.setLevel(level)
            log_fmt = "%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s"
            fh_formatter = Formatter(log_fmt)
            fh.setFormatter(fh_formatter)
            log.addHandler(fh)
        return log

    @classmethod
    def argparse_config(cls, args: argparse.Namespace, unknown: list[str] = []):
        assert len(args.exp_name or "") > 0, "Specify `exp_name`."
        assert len(args.cfg_dir or "") and Path(args.cfg_dir).exists(), f"--cfg_dir={args.cfg_dir} does not exist."

        cfg_dir = Path(args.cfg_dir)
        initialize_config_dir(version_base=None, config_dir=str(cfg_dir.absolute()), job_name=args.exp_name)
        unknown_pplus = ["++" + a.split("+")[-1] for a in unknown]
        base_config: DictConfig = compose(config_name=args.cfg_name, overrides=unknown_pplus)
        print(f"Loaded config from '{args.cfg_name}.yaml'")
        print(f"Additional args from command line: {unknown_pplus}")

        cfg = cls(EXP_NAME=args.exp_name, IS_TEST=args.test)
        merge_config_list = [base_config]
        for _ in range(10):
            if len(merge_config_list) < 1:
                break
            current = merge_config_list[-1]
            parent_name = current.PARENT_CFG_NAME
            if len(parent_name or "") == 0:
                break
            parent_cfg: DictConfig = compose(config_name=parent_name)
            merge_config_list.append(parent_cfg)
        else:
            raise RuntimeError("Too much recursion of PARENT_CFG_NAME. Maximum 10.")
        for parent_cfg in reversed(merge_config_list):
            if len(cfg.PARENT_CFG_NAME or "") > 0 and args.verbose:
                print(f"Merging config from '{cfg.PARENT_CFG_NAME}.yaml'")
            cfg.merge_from(parent_cfg)

        assert cfg.EXP_NAME == args.EXP_NAME, f"Do not specify EXP_NAME inside {args.cfg_dir}, instead as cli option."
        return cfg

    @classmethod
    def argparse(cls, _args: Optional[list[str]] = None, _unknown: Optional[list[str]] = None):
        args = parser.parse_args(_args) if _args is not None else default_args()
        unknown = default_unknown() + (_unknown or [])
        self = cls.argparse_config(args, unknown)
        self.argparse_manager(args)
        if self.DISTRIBUTED and not self._show_and_exit:
            self.torch_setup_on_distributed(args)
        return self

    def print_whole_config(self):
        train_or_test = "test" if self.IS_TEST else "train"
        self.log.info(f"==== [red]{self.EXP_NAME}[/]: {train_or_test} -> '{self.RESULT_DIR}/' ====")
        self.print_config()
        if self._show_and_exit:
            sys.exit(0)

    @property
    def log(self):
        if self._logger is None:
            self._logger = self.setup_logger(self.log_level, self._use_rich, self._log_file)
        return self._logger

    @property
    def debug(self):
        return self.log.debug

    @property
    def info(self):
        return self.log.info

    @property
    def warn(self):
        return self.log.warn

    @property
    def error(self):
        return self.log.error
