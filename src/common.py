import logging

import numpy as np
import torch
from game_2048_3_3 import State

import config_2048 as cfg

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


def max_mov(canmov: list[bool], bd: State, value: list):
    cfg.MODEL.eval()  # モデルを評価モードに設定
    inputs = torch.zeros(4, 99, device="cpu")
    sub_list = []
    for i in range(4):
        copy_bd = bd.clone()
        copy_bd.play(i)
        sub_list.append(copy_bd.score - bd.score)
        write_make_input(copy_bd.board, inputs[i, :])
    inputs = inputs.to(cfg.DEVICE)
    tmp: torch.Tensor = cfg.MODEL.forward(inputs)
    debug_list = []
    for i in range(4):
        if canmov[i]:
            # log.debug(f"can move to {i}")
            value[i] = float(tmp.data[i]) + sub_list[i]
            debug_list.append({"sub": sub_list[i], "bd": copy_bd.board})
            # print(model.forward(input))
            # log.debug(f"評価値 : {value[i]}")
        else:
            value[i] = -1e10
            tmp.data[i] = -1e10
    logger.debug(f"{debug_list=}")
    max_index = np.argmax(value)
    return max_index, value[max_index], tmp.data
