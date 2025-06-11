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
from common import max_mov, write_make_input
from config_2048 import MODEL_1, MODEL_2
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
optimizer = optim.Adam(cfg.MODEL.parameters(), lr=0.001)  # 最適化アルゴリズムの選択
logger = logging.getLogger(__name__)
MODEL_TYPE = 1


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


def _train(target_model: torch.nn.Module, inputs, targets):
    """
    モデルを学習させる関数
    :param target_model: 学習対象のモデル
    :param inputs: 入力データ
    :param targets: ターゲットデータ
    """
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
def train(records, count: int = 1):
    # inputsには盤面の情報、targetsには評価値が入る
    targets_1 = []
    targets_2 = []
    inputs_1 = []
    inputs_2 = []
    for record in records:
        board, value, model_type = record
        if model_type == 1:
            inputs_1.append(board)
            targets_1.append(value)
        elif model_type == 2:
            inputs_2.append(board)
            targets_2.append(value)

    # お互いのモデルに対して学習を行う
    if inputs_1:
        _train(MODEL_2, inputs_1, targets_1)
        logger.info(f"train [model1] {count * len(targets_1)} board")
    if inputs_2:
        _train(MODEL_1, inputs_2, targets_2)
        logger.info(f"train [model2] {count * len(targets_2)} board")


def put_queue(board: np.ndarray, value: torch.Tensor, model_type: int):
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
                train_queue.put((bd, value, model_type))
        elif args.sym_type == 1:
            rnd.shuffle(bd_list)
            rand_bd = bd_list.pop()
            logger.debug(f"{bd_list=}\n\t{rand_bd=}")
            train_queue.put((rand_bd, value, model_type))
        elif args.sym_type == 2:
            unique_bd = set(tuple(bd) for bd in bd_list)
            logger.debug(f"{bd_list=}\n\t{unique_bd=}")
            for bd in unique_bd:
                train_queue.put((np.array(bd), value, model_type))
        elif args.sym_type == 3:
            # 自分自身を選択する
            board_index = 0
            target = bd_list[board_index]
            logger.debug(f"{bd_list=}\n\t{target=}")
            train_queue.put((target, value, model_type))
        elif args.sym_type == 4:
            # 0, 2を12時間回しましょう
            board_index = 2
            target = bd_list[board_index]
            logger.debug(f"{bd_list=}\n\t{target=}")
            train_queue.put((target, value, model_type))
    else:
        train_queue.put((board, value, model_type))


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
                    if args.ddqn == 0:
                        model_type = 1 if games % 2 == 0 else 2
                    elif args.ddqn == 1:
                        model_type = rnd.choice([1, 2])
                    elif args.ddqn == 2:
                        global MODEL_TYPE
                        MODEL_TYPE = 1 if MODEL_TYPE == 2 else 2
                        model_type = MODEL_TYPE
                    turn += 1
                    canmov = [bd.canMoveTo(i) for i in range(4)]
                    value = [0.0 for _ in range(4)]
                    copy_bd = bd.clone()
                    max_index, max_value, _ = max_mov(
                        canmov, copy_bd, value, model_type
                    )
                    bd.play(max_index)
                    if last_board is not None:
                        put_queue(last_board.copy(), max_value, model_type)
                    last_board = bd.clone().board
                    if turn == 1:
                        init_eval_1 = get_eval(bd.board, 1)
                        init_eval_2 = get_eval(bd.board, 2)
                    bd.putNewTile()
                    states.append(bd.clone())
                    if bd.isGameOver():
                        board_print(bd)
                        put_queue(last_board.copy(), torch.tensor(0), model_type)
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


def get_eval(board: np.ndarray, model_type: int):
    if model_type == 1:
        model = MODEL_1
    elif model_type == 2:
        model = MODEL_2
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
        train(records, train_count)
        records.clear()

    model_path = cfg.MODEL_DIR / f"{cfg.LOG_PATH.stem}_{train_count}.pth"
    initial_board = [0, 0, 0, 0, 0, 1, 0, 0, 1]
    for i in [1, 2]:
        model = MODEL_1 if i == 1 else MODEL_2
        torch.save(model.state_dict(), model_path.with_stem(f"[{i}]_{model_path.stem}"))
        logger.info(f"save {model_path.name} {train_count=}")
        logger.info(f"評価値: {get_eval(initial_board, i)}")
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
