import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue

import numpy as np
import torch
from numpy import random as rnd
from torch import nn, optim

import config_2048 as cfg
from arg import args
from common import get_values, write_make_input
from config_2048 import MAIN_NETWORK, TARGET_NETWORK
from game_2048_3_3 import State

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
tasks = os.cpu_count()
train_queue = Queue(4096)  # 学習データをスレッドから受け取るキュー
stop_event = threading.Event()  # スレッド終了用に使用
bat_size = 1024  # バッチサイズ
criterion = nn.MSELoss()  # 損失関数
optimizer_main = optim.Adam(MAIN_NETWORK.parameters(), lr=0.001)
optimizer_target = optim.Adam(TARGET_NETWORK.parameters(), lr=0.001)
logger = logging.getLogger(__name__)


# デバッグ用の関数
def board_print(bd: State, level=logging.DEBUG):
    for i in range(3):
        logger.log(
            level, f"{bd.board[i * 3 + 0]} {bd.board[i * 3 + 1]} {bd.board[i * 3 + 2]}"
        )


def rotate_board(board: np.ndarray):
    """
    盤面を90度回転させる（NumPy操作で最適化）
    """
    board_2d = board.reshape(3, 3)
    rotated_2d = np.rot90(board_2d, k=-1)  # 時計回りに90度回転
    return rotated_2d.flatten()


def mirror_board(board: np.ndarray):
    """
    盤面を左右反転させる（NumPy操作で最適化）
    """
    board_2d = board.reshape(3, 3)
    mirrored_2d = np.fliplr(board_2d)  # 左右反転
    return mirrored_2d.flatten()


def _train(model_type: int, inputs, targets):
    """入力とターゲットを使用してモデルを学習する関数。

    Args:
        model_type (int): 1ならメインネットワーク、2ならターゲットネットワークを使用
        inputs (list[np.ndarray]): 盤面の情報を含むテンソル
        targets (list[float] or np.ndarray): 評価値を含むテンソル
    """
    target_model = MAIN_NETWORK if model_type == 1 else TARGET_NETWORK
    optimizer = optimizer_main if model_type == 1 else optimizer_target

    target_model.train()  # モデルを学習モードに設定
    optimizer.zero_grad()  # 勾配をゼロに初期化
    tmp = torch.zeros(len(inputs), 99, device="cpu")
    for i in range(len(inputs)):
        write_make_input(inputs[i], tmp[i, :])
    inputs = tmp.to(cfg.DEVICE)
    # ネットワークにデータを入力し、順伝播を行う
    outputs = target_model.forward(inputs)
    targets = torch.as_tensor(targets, dtype=torch.float32)
    targets = targets.reshape(-1, 1)  # ターゲットの形状を調整
    targets = targets.to(cfg.DEVICE)
    loss = criterion(outputs, targets)  # 損失を計算
    loss.backward()  # 逆伝播を行い、各パラメータの勾配を計算
    optimizer.step()
    logger.debug(f"loss : {loss.item()}")


# 学習用の関数
def train(records: list[dict], count: int = 1):
    # inputsには盤面の情報、targetsには評価値が入る
    main_values = []
    target_values = []
    boards = []
    for record in records:
        board = record["board"]
        main_value = record["main_value"]
        target_value = record["target_value"]
        boards.append(board)
        main_values.append(main_value)
        target_values.append(target_value)

    # メインネットワークはターゲットから得られた評価値を使用して学習する
    logger.debug(
        f"train {count=}, {len(boards)=}, {len(main_values)=}, {len(target_values)=}"
    )
    if len(boards) == 0:
        logger.warning("No records to train.")
        return
    if len(boards) != len(main_values) or len(boards) != len(target_values):
        logger.error(
            f"Length mismatch: {len(boards)=}, {len(main_values)=}, {len(target_values)=}"
        )
        return
    try:
        _train(1, boards, target_values)
        _train(2, boards, main_values)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return


def put_queue(board: np.ndarray, main_value: float, target_value: float):
    board_cp = board.copy()
    if args.symmetry:
        bd_list: list[np.ndarray] = [board]
        for _ in range(3):
            board = rotate_board(board)
            bd_list.append(board)
        board = mirror_board(board)
        bd_list.append(board)
        for _ in range(3):
            board = rotate_board(board)
            bd_list.append(board)

        if args.sym_type == 0:
            logger.debug(f"{bd_list=}\n\t{board_cp=}")
            for bd in bd_list:
                train_queue.put(
                    {
                        "board": bd,
                        "main_value": main_value,
                        "target_value": target_value,
                    }
                )
        elif args.sym_type == 1:
            rnd.shuffle(bd_list)
            rand_bd = bd_list.pop()
            logger.debug(f"{bd_list=}\n\t{rand_bd=}")
            train_queue.put(
                {
                    "board": rand_bd,
                    "main_value": main_value,
                    "target_value": target_value,
                }
            )
        elif args.sym_type == 2:
            unique_bd = set(tuple(bd) for bd in bd_list)
            logger.debug(f"{bd_list=}\n\t{unique_bd=}")
            for bd in unique_bd:
                train_queue.put(
                    {
                        "board": np.array(bd),
                        "main_value": main_value,
                        "target_value": target_value,
                    }
                )
        elif args.sym_type == 3:
            # 自分自身を選択する
            board_index = 0
            target = bd_list[board_index]
            logger.debug(f"{bd_list=}\n\t{target=}")
            train_queue.put(
                {
                    "board": target,
                    "main_value": main_value,
                    "target_value": target_value,
                }
            )
        elif args.sym_type == 4:
            # 0, 2を12時間回しましょう
            board_index = 2
            target = bd_list[board_index]
            logger.debug(f"{bd_list=}\n\t{target=}")
            train_queue.put(
                {
                    "board": target,
                    "main_value": main_value,
                    "target_value": target_value,
                }
            )
    else:
        train_queue.put(
            {
                "board": board,
                "main_value": main_value,
                "target_value": target_value,
            }
        )


def play_game(thread_id: int):
    try:
        games = 0
        while not stop_event.is_set():
            games += 1
            states = []
            bd = State()
            bd.initGame()
            turn = 0
            for count in range(10_000):
                last_board = None
                init_eval_1 = 0
                init_eval_2 = 0
                while not stop_event.is_set():
                    turn += 1
                    canmov = [bd.canMoveTo(i) for i in range(4)]
                    copy_bd = bd.clone()
                    main_values, target_values = get_values(canmov, copy_bd)
                    # main_valuesから最大の評価値を持つインデックスを取得
                    main_max_index = np.argmax(main_values)
                    target_max_value = np.max(target_values)
                    bd.play(main_max_index)
                    if last_board is not None:
                        put_queue(
                            last_board.copy(),
                            main_value=main_values[main_max_index],
                            target_value=target_max_value,
                        )
                    last_board = bd.clone().board
                    if turn == 1:
                        init_eval_1 = get_eval(bd.board, MAIN_NETWORK)
                        init_eval_2 = get_eval(bd.board, TARGET_NETWORK)
                    bd.putNewTile()
                    states.append(bd.clone())
                    if bd.isGameOver():
                        board_print(bd)
                        put_queue(last_board.copy(), torch.tensor(0), torch.tensor(0))
                        logger.info(
                            f"GAMEOVER: {thread_id=:02d} {count=:03d} {bd.score=:04d} {turn=:04d} queue_size={train_queue.qsize():04d} {init_eval_1=:.2f} {init_eval_2=:.2f}"
                        )
                        break
                if args.restart and len(states) > 10:
                    bd = states[len(states) // 2]
                    turn -= len(states) // 2
                    states = []
                else:
                    break
    except Exception as e:
        logger.exception(e)
        stop_event.set()
        while train_queue.qsize() > 0:
            train_queue.get()


def get_eval(board: np.ndarray, model: torch.nn.Module):
    x = torch.zeros(1, 99, device=cfg.DEVICE)
    write_make_input(board, x[0, :])
    model.eval()  # モデルを評価モードに設定
    eval = model.forward(x)
    return eval.item()


def main():
    train_count = 0
    records = []
    executor = ThreadPoolExecutor(max_workers=tasks)
    for i in range(tasks):
        executor.submit(play_game, i)
    start_time = datetime.now()
    while datetime.now() - start_time < cfg.TIME_LIMIT:
        train_count += 1
        while len(records) != bat_size:
            records.append(train_queue.get())
        logger.info(
            f"train {train_count=}, {len(records)=}, queue_size={train_queue.qsize()}"
        )
        train(records, train_count)
        records.clear()

    model_path = cfg.MODEL_DIR / f"{cfg.LOG_PATH.stem}_{train_count}.pth"
    initial_board = [0, 0, 0, 0, 0, 1, 0, 0, 1]
    for i in [1, 2]:
        model = MAIN_NETWORK if i == 1 else TARGET_NETWORK
        torch.save(model.state_dict(), model_path.with_stem(f"[{i}]_{model_path.stem}"))
        logger.info(f"save {model_path.name} {train_count=}")
        logger.info(f"評価値: {get_eval(initial_board, model)}")
    stop_event.set()
    logger.info("stop event set")
    while train_queue.qsize() > 0:
        records.append(train_queue.get())
    executor.shutdown()
    logger.info("All threads have been successfully terminated.")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e
