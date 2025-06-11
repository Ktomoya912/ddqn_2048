import logging
import re
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import torch
from rich.logging import RichHandler

from arg import args, game_conf

logger = logging.getLogger(__name__)

try:
    exec(f"from models import {args.model} as modeler")
    exec(f"from models import {args.model} as modeler2")
except ImportError as e:
    logger.error(f"Model {args.model} not found in models module: {e}")
    raise ImportError(
        f"Model {args.model} not found. Please check the models module."
    ) from e


def get_trained_model(log_path: Path, device) -> OrderedDict:
    pat = r"\d{8}T\d{6}_"
    config_stem = re.sub(pat, "", log_path.stem)
    target = [
        model_file
        for model_file in Path("model").glob("*.pth")
        if re.sub(pat, "", model_file.stem).startswith(config_stem)
    ]
    if not target:
        logger.warning(f"Model file not found: {config_stem}")
        return
    target.sort()
    state_dict = torch.load(target[-1], map_location=device, weights_only=True)
    logger.info(f"Model loaded: {target[-1]}")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def get_model_name():
    # game_confからモデル名を取得
    models_names = []
    for k, v in game_conf.items():
        models_names.append(f"[{k}-{v}]")
    return "".join(models_names)


start_time = datetime.now()
TRAIN_COUNT = args.train_count
TIME_LIMIT = timedelta(hours=args.hours) if args.hours > 0 else timedelta(days=1)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
LOG_PATH = Path(f"log/{start_time.strftime('%Y%m%dT%H%M%S')}_{get_model_name()}.log")
LOG_PATH.parent.mkdir(exist_ok=True)
FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
logging.basicConfig(
    level=args.log,
    format=FORMAT,
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        RichHandler(show_time=False, show_level=False, show_path=False),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("rich")
for key, value in args._get_kwargs():
    logger.info(f"{key} : {value}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
MODEL_1: torch.nn.Module = modeler.Model().to(DEVICE)  # noqa: F821
MODEL_2: torch.nn.Module = modeler2.Model().to(DEVICE)  # noqa: F821
if args.load_model:
    if new_state_dict := get_trained_model(LOG_PATH, DEVICE):
        MODEL_1.load_state_dict(new_state_dict)
if torch.cuda.is_available():
    MODEL_1 = torch.nn.DataParallel(MODEL_1)
    MODEL_2 = torch.nn.DataParallel(MODEL_2)
