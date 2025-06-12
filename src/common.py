import logging

import numpy as np
import torch

from config_2048 import DEVICE, MAIN_NETWORK, TARGET_NETWORK
from game_2048_3_3 import State

logger = logging.getLogger(__name__)


def calc_progress(board: np.ndarray):
    """
    ボード状態から進捗度（progress）を計算
    各タイルの値の2の累乗和を2で割った値を返す。
    """
    return sum(2**i for i in board if i > 0) // 2


def write_make_input(board: np.ndarray, x: torch.Tensor):
    for j in range(9):
        x[board[j] * 9 + j] = 1


def get_values(canmov: list[bool], bd: State):
    """移動可能な方向の評価値を計算する。
    移動可能な方向に対して、評価値を計算し、最大の評価値とその方向を返す。

    Args:
        canmov (list[bool]): 移動可能な方向のリスト
        bd (State): ゲームの状態を表すStateオブジェクト

    Returns:
        tuple: メインネットワークの評価値、ターゲットネットワークの評価値
    """
    MAIN_NETWORK.eval()
    TARGET_NETWORK.eval()
    inputs = torch.zeros(4, 99, device="cpu")
    sub_list = []
    for i in range(4):
        copy_bd = bd.clone()
        copy_bd.play(i)
        sub_list.append(copy_bd.score - bd.score)
        write_make_input(copy_bd.board, inputs[i, :])
    inputs = inputs.to(DEVICE)
    main_result: torch.Tensor = MAIN_NETWORK.forward(inputs)
    target_result: torch.Tensor = TARGET_NETWORK.forward(inputs)
    main_value = [0.0] * 4
    target_value = [0.0] * 4
    for i in range(4):
        if canmov[i]:
            main_value[i] = float(main_result.data[i]) + sub_list[i]
            target_value[i] = float(target_result.data[i]) + sub_list[i]
        else:
            main_value[i] = -1e10
            main_result.data[i] = -1e10
    return main_value, target_value
