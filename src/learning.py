import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue

import numpy as np
import torch
from game_2048_3_3 import State
from numpy import random as rnd
from torch import nn, optim

import config_2048 as cfg
from arg import args
from common import max_mov, write_make_input

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
tasks = os.cpu_count()
train_queue = Queue(1024)  # 学習データをスレッドから受け取るキュー
stop_event = threading.Event()  # スレッド終了用に使用

criterion = nn.MSELoss()  # 損失関数
optimizer = optim.Adam(cfg.MODEL.parameters(), lr=0.001)  # 最適化アルゴリズムの選択
logger = logging.getLogger(__name__)


# デバッグ用の関数
def board_print(bd: State, level=logging.DEBUG):
    for i in range(3):
        logger.log(
            level, f"{bd.board[i * 3 + 0]} {bd.board[i * 3 + 1]} {bd.board[i * 3 + 2]}"
        )


# ランダムに手を選択する用の関数(使ってない)
def random_mov(canmov):
    index = random.choice(canmov)
    return index


def rotate_board(board: np.ndarray):
    """
    盤面を90度回転させる
    """
    new_board = np.zeros(9, dtype=np.int64)
    for i in range(3):
        for j in range(3):
            new_board[i * 3 + j] = board[(2 - j) * 3 + i]
    return new_board


def mirror_board(board: np.ndarray):
    """
    盤面を反転させる
    """
    new_board = np.zeros(9, dtype=np.int64)
    for i in range(3):
        for j in range(3):
            new_board[i * 3 + j] = board[i * 3 + (2 - j)]
    return new_board


# 学習用の関数
def train(inputs: list, targets: list):
    # inputsには盤面の情報、targetsには評価値が入る
    cfg.MODEL.train()  # モデルを学習モードに設定
    optimizer.zero_grad()  # 勾配をゼロに初期化
    tmp = torch.zeros(len(inputs), 99, device="cpu")
    for i in range(len(inputs)):
        write_make_input(inputs[i], tmp[i, :])
    inputs = tmp.to(cfg.DEVICE)
    # ネットワークにデータを入力し、順伝播を行う
    outputs = cfg.MODEL.forward(inputs)
    targets = torch.as_tensor(targets, dtype=torch.float32)
    targets = targets.reshape(-1, 1)  # 2次元に変換
    targets = targets.to(cfg.DEVICE)
    loss = criterion(outputs, targets)  # 損失を計算
    loss.backward()  # 逆伝播を行い、各パラメータの勾配を計算
    optimizer.step()
    logger.debug(f"loss : {loss.item()}")


def put_queue(board: np.ndarray, value: torch.Tensor):
    board_cp = board.copy()
    if args.symmetry:
        bd_list: list[np.ndarray] = []
        bd_list.append(board)
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
                train_queue.put((bd, value))
        elif args.sym_type == 1:
            rnd.shuffle(bd_list)
            rand_bd = bd_list.pop()
            logger.debug(f"{bd_list=}\n\t{rand_bd=}")
            train_queue.put((rand_bd, value))
        elif args.sym_type == 2:
            unique_bd = set(tuple(bd) for bd in bd_list)
            logger.debug(f"{bd_list=}\n\t{unique_bd=}")
            for bd in unique_bd:
                train_queue.put((np.array(bd), value))
        elif args.sym_type == 3:
            # 自分自身を選択する
            # board_index = next(
            #     i for i, b in enumerate(bd_list) if np.array_equal(b, board_cp)
            # )
            # 0, 2を12時間回しましょう
            board_index = 0
            target = bd_list[board_index]
            logger.debug(f"{bd_list=}\n\t{target=}")
            train_queue.put((target, value))
        elif args.sym_type == 4:
            # 0, 2を12時間回しましょう
            board_index = 2
            target = bd_list[board_index]
            logger.debug(f"{bd_list=}\n\t{target=}")
            train_queue.put((target, value))
    else:
        # tmp = board.copy()
        # for _ in range(4):
        #     board = rotate_board(board)
        # if sum((tmp - board) ** 2) != 0:
        #     logger.error(
        #         f"違いますよ！！！！！！！！！！！！！！！！！！！！！！！！！ {tmp=}, {board=}"
        #     )
        train_queue.put((board, value))


def play_game(thread_id: int):
    try:
        while not stop_event.is_set():
            states = []
            bd = State()
            bd.initGame()
            turn = 0
            for count in range(10_000):
                last_board = None
                init_eval = 0
                while not stop_event.is_set():
                    turn += 1
                    canmov = [bd.canMoveTo(i) for i in range(4)]
                    value = [0.0 for _ in range(4)]
                    copy_bd = bd.clone()
                    max_index, max_value, _ = max_mov(canmov, copy_bd, value)
                    bd.play(max_index)
                    if last_board is not None:
                        put_queue(last_board.copy(), max_value)
                    last_board = bd.clone().board
                    if turn == 1:
                        init_eval = get_eval(bd.board)
                    bd.putNewTile()
                    states.append(bd.clone())
                    if bd.isGameOver():
                        board_print(bd)
                        put_queue(last_board.copy(), torch.tensor(0))
                        logger.info(
                            f"GAMEOVER: {thread_id=:02d} {count=:03d} {bd.score=:04d} {turn=:04d} queue_size={train_queue.qsize()} {init_eval=:.2f}"
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


def get_eval(board=[0, 1, 1, 0, 0, 0, 0, 0, 0]):
    x = torch.zeros(1, 99, device=cfg.DEVICE)
    write_make_input(board, x[0, :])
    cfg.MODEL.eval()  # モデルを評価モードに設定
    eval = cfg.MODEL.forward(x)
    # logger.info(f"評価値 : {cfg.MODEL.forward(x)}")
    return eval.item()


def main():
    train_count = 0
    train_data = []
    target_data = []
    executor = ThreadPoolExecutor(max_workers=tasks)
    for i in range(tasks):
        executor.submit(play_game, i)
    start_time = datetime.now()
    while True:
        train_count += 1
        while len(train_data) != 1024:
            board, value = train_queue.get()
            train_data.append(board)
            target_data.append(value)
        train(train_data, target_data)
        logger.info(f"train {train_count * len(train_data)} board")
        train_data.clear()
        target_data.clear()
        if (
            datetime.now() - start_time > cfg.TIME_LIMIT
            or train_count == cfg.TRAIN_COUNT
        ):
            model_path = cfg.MODEL_DIR / f"{cfg.LOG_PATH.stem}_{train_count}.pth"
            torch.save(cfg.MODEL.state_dict(), model_path)
            logger.info(f"save {model_path.name} {train_count=}")
            logger.info(f"評価値: {get_eval()}")
            break
    stop_event.set()
    logger.info("stop event set")
    while train_queue.qsize() > 0:
        train_data.append(train_queue.get())
    executor.shutdown()
    logger.info("All threads have been successfully terminated.")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e
