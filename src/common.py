import logging

import numpy as np
import torch

from config_2048 import DEVICE, LOG_PATH, MAIN_NETWORK, MODEL_DIR, TARGET_NETWORK
from game_2048_3_3 import State

logger = logging.getLogger(__name__)


def save_models(save_count: int = -1):
    main_model_path = MODEL_DIR / f"main_{save_count}_{LOG_PATH.stem}.pth"
    target_model_path = MODEL_DIR / f"target_{save_count}_{LOG_PATH.stem}.pth"

    torch.save(MAIN_NETWORK.state_dict(), main_model_path)
    logger.info(f"save {main_model_path.name} {save_count=}")
    torch.save(TARGET_NETWORK.state_dict(), target_model_path)
    logger.info(f"save {target_model_path.name} {save_count=}")


def calc_progress(board: np.ndarray):
    """
    ボード状態から進捗度（progress）を計算
    各タイルの値の2の累乗和を2で割った値を返す。
    """
    return sum(2**i for i in board if i > 0) // 2


def write_make_input(board: np.ndarray, x: torch.Tensor):
    for j in range(9):
        x[board[j] * 9 + j] = 1


def get_values(canmov: list[bool], bd: State, packs: list):
    """移動可能な方向の評価値を計算する。
    移動可能な方向に対して、評価値を計算し、最大の評価値とその方向を返す。

    Args:
        canmov (list[bool]): 移動可能な方向のリスト
        bd (State): ゲームの状態を表すStateオブジェクト

    Returns:
        tuple: メインネットワークの評価値、ターゲットネットワークの評価値
    """
    packs[0]["model"].eval()
    packs[1]["model"].eval()
    inputs = torch.zeros(4, 99, device="cpu")
    sub_list = []
    for i in range(4):
        copy_bd = bd.clone()
        copy_bd.play(i)
        sub_list.append(copy_bd.score - bd.score)
        write_make_input(copy_bd.board, inputs[i, :])
    inputs = inputs.to(DEVICE)
    main_result: torch.Tensor = packs[0]["model"].forward(inputs)
    target_result: torch.Tensor = packs[1]["model"].forward(inputs)
    main_value = [-1e10] * 4
    target_value = [-1e10] * 4
    for i in range(4):
        if canmov[i]:
            main_value[i] = float(main_result.data[i]) + sub_list[i]
            target_value[i] = float(target_result.data[i]) + sub_list[i]

    return main_value, target_value


def get_one_values(canmov: list[bool], bd: State, model: torch.nn.Module):
    """移動可能な方向の評価値を計算する。
    移動可能な方向に対して、評価値を計算し、最大の評価値とその方向を返す。

    Args:
        canmov (list[bool]): 移動可能な方向のリスト
        bd (State): ゲームの状態を表すStateオブジェクト
        model (torch.nn.Module): 評価を行うニューラルネットワークモデル

    Returns:
        list: 各方向の評価値のリスト
    """
    model.eval()
    inputs = torch.zeros(4, 99, device="cpu")
    sub_list = []
    for i in range(4):
        copy_bd = bd.clone()
        copy_bd.play(i)
        sub_list.append(copy_bd.score - bd.score)
        write_make_input(copy_bd.board, inputs[i, :])
    inputs = inputs.to(DEVICE)
    result: torch.Tensor = model.forward(inputs)
    values = [-1e10] * 4
    for i in range(4):
        if canmov[i]:
            values[i] = float(result.data[i]) + sub_list[i]

    return values
