import logging
import re
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import torch

from arg import args, game_conf

logger = logging.getLogger(__name__)
loaded_model = ""

try:
    exec(f"from models import {args.model} as modeler")
    exec(f"from models import {args.model} as modeler2")
except ImportError as e:
    logger.error(f"Model {args.model} not found in models module: {e}")
    raise ImportError(
        f"Model {args.model} not found. Please check the models module."
    ) from e


def get_trained_model(log_path: Path, device, type_: str) -> OrderedDict:
    pat = r"(\w+_)*(\-?\d*_)?(\[.*\]_)*\d{8}T\d{6}_"
    config_stem = re.sub(pat, "", log_path.stem)
    target = [
        model_file
        for model_file in MODEL_DIR.glob("*.pth")
        if re.sub(pat, "", model_file.stem).endswith(config_stem)
        and model_file.stem.startswith(type_)
    ]
    if not target:
        logger.warning(f"Model file not found: {config_stem}")
        return
    target.sort()
    if args.load_script:
        target = [
            model_file for model_file in target if args.load_script in model_file.name
        ]
        if not target:
            logger.warning(
                f"Model file with script {args.load_script} not found: {config_stem}"
            )
            return
    state_dict = torch.load(target[-1], map_location=device, weights_only=True)
    logger.info(f"Model loaded: {target[-1]}")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict, target[-1].stem.replace(f"{type_}_", "")


def get_model_name():
    # game_confからモデル名を取得
    models_names = []
    for k, v in game_conf.items():
        models_names.append(f"[{k}-{v}]")
    return "".join(models_names)


script_name = Path(sys.argv[0]).name
start_time = datetime.now()
TIME_LIMIT = timedelta(hours=args.hours) if args.hours > 0 else timedelta(days=1)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
LOG_PATH = Path(f"log/{start_time.strftime('%Y%m%dT%H%M%S')}_{get_model_name()}.log")
if script_name != "learning.py":
    LOG_PATH = LOG_PATH.with_name(f"[{script_name}]_{LOG_PATH.name}")
LOG_PATH.parent.mkdir(exist_ok=True)
FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
logging.basicConfig(
    level=args.log,
    format=FORMAT,
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        # RichHandler(show_time=False, show_level=False, show_path=False),
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("config_2048")
for key, value in args._get_kwargs():
    logger.info(f"{key} : {value}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
MAIN_NETWORK: torch.nn.Module = modeler.Model().to(DEVICE)  # noqa: F821
TARGET_NETWORK: torch.nn.Module = modeler2.Model().to(DEVICE)  # noqa: F821
if args.load_model:
    if data := get_trained_model(LOG_PATH, DEVICE, "main"):
        MAIN_NETWORK.load_state_dict(data[0])
        loaded_model = data[1]
    if data := get_trained_model(LOG_PATH, DEVICE, "target"):
        TARGET_NETWORK.load_state_dict(data[0])
        loaded_model = data[1]
