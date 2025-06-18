import logging
import os
import random
import threading
import time
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
stop_event = threading.Event()  # スレッド終了用に使用
bat_size = 1024  # バッチサイズ
criterion = nn.MSELoss()  # 損失関数
optimizer_main = optim.Adam(MAIN_NETWORK.parameters(), lr=0.001)
optimizer_target = optim.Adam(TARGET_NETWORK.parameters(), lr=0.001)
pack_main = {
    "model": MAIN_NETWORK,
    "optimizer": optimizer_main,
    "name": "main",
    "queue": Queue(100000),
}
pack_target = {
    "model": TARGET_NETWORK,
    "optimizer": optimizer_target,
    "name": "target",
    "queue": Queue(100000),
}
# ゲームごとに貯めて学習
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


# 学習用の関数
def train(records: list[dict], pack: dict, count: int = 1):
    # inputsには盤面の情報、targetsには評価値が入る
    target_values = []
    boards = []
    for record in records:
        board = record["board"]
        target_value = record["target_value"]
        boards.append(board)
        target_values.append(target_value)

    # メインネットワークはターゲットから得られた評価値を使用して学習する
    logger.info(f"train {count=}, {len(boards)=}, {len(target_values)=}")
    if len(boards) == 0:
        logger.warning("No records to train.")
        return
    if len(boards) != len(target_values):
        logger.error(f"Length mismatch: {len(boards)=}, {len(target_values)=}")
        return

    model = pack["model"]
    optimizer = pack["optimizer"]

    model.train()  # モデルを学習モードに設定
    optimizer.zero_grad()  # 勾配をゼロに初期化
    tmp = torch.zeros(len(boards), 99, device="cpu")
    for i in range(len(boards)):
        write_make_input(boards[i], tmp[i, :])
    inputs = tmp.to(cfg.DEVICE)
    # ネットワークにデータを入力し、順伝播を行う
    outputs = model.forward(inputs)
    targets = torch.as_tensor(target_values, dtype=torch.float32)
    targets = targets.reshape(-1, 1)  # ターゲットの形状を調整
    targets = targets.to(cfg.DEVICE)
    loss = criterion(outputs, targets)  # 損失を計算
    loss.backward()  # 逆伝播を行い、各パラメータの勾配を計算
    optimizer.step()
    logger.debug(f"loss : {loss.item()}")


def put_queue(board: np.ndarray, main_value: float, target_value: float, packs):
    board_cp = board.copy()
    queue = packs[0]["queue"]
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
                queue.put(
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
            queue.put(
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
                queue.put(
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
            queue.put(
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
            queue.put(
                {
                    "board": target,
                    "main_value": main_value,
                    "target_value": target_value,
                }
            )
    else:
        queue.put(
            {
                "board": board,
                "main_value": main_value,
                "target_value": target_value,
            }
        )


def play_game(thread_id: int):
    packs = [pack_main, pack_target]
    try:
        games = 0
        while not stop_event.is_set():
            games += 1
            states = []
            bd = State()
            bd.initGame()
            turn = 0
            packs.reverse()
            for count in range(10_000):
                last_board = None
                init_eval_1 = 0
                init_eval_2 = 0
                while not stop_event.is_set():
                    turn += 1
                    canmov = [bd.canMoveTo(i) for i in range(4)]
                    copy_bd = bd.clone()
                    main_values, target_values = get_values(canmov, copy_bd, packs)
                    # main_valuesから最大の評価値を持つインデックスを取得
                    main_max_index = np.argmax(main_values)
                    bd.play(main_max_index)
                    if last_board is not None:
                        put_queue(
                            last_board.copy(),
                            main_value=main_values[main_max_index],
                            target_value=target_values[main_max_index],
                            packs=packs,
                        )
                    last_board = bd.clone().board
                    if turn == 1:
                        init_eval_1 = get_eval(bd.board, MAIN_NETWORK)
                        init_eval_2 = get_eval(bd.board, TARGET_NETWORK)
                    bd.putNewTile()
                    states.append(bd.clone())
                    if bd.isGameOver():
                        board_print(bd)
                        put_queue(
                            last_board.copy(),
                            torch.tensor(0),
                            torch.tensor(0),
                            packs=packs,
                        )
                        logger.info(
                            f'GAMEOVER: {thread_id=:02d} {count=:03d} {bd.score=:04d} {turn=:04d} {packs[0]["name"].ljust(7)}queue_size={packs[0]["queue"].qsize():04d} {init_eval_1=:.2f} {init_eval_2=:.2f}'
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


def get_eval(board: np.ndarray, model: torch.nn.Module):
    x = torch.zeros(1, 99, device=cfg.DEVICE)
    write_make_input(board, x[0, :])
    model.eval()  # モデルを評価モードに設定
    eval = model.forward(x)
    return eval.item()


def batch_trainer(pack: dict):
    train_count = 0
    records = []
    while not stop_event.is_set():
        train_count += 1
        while len(records) != bat_size:
            records.append(pack["queue"].get())
        train(records, pack, train_count)

        records.clear()
    return train_count


def clear_queues():
    while pack_main["queue"].qsize() > 0 and pack_target["queue"].qsize() > 0:
        pack_main["queue"].get()
        pack_target["queue"].get()
    logger.info("Queues cleared, stopping threads...")


def save_models():
    main_model_path = cfg.MODEL_DIR / f"main_{cfg.LOG_PATH.stem}_toggle.pth"
    target_model_path = cfg.MODEL_DIR / f"target_{cfg.LOG_PATH.stem}_toggle.pth"

    torch.save(MAIN_NETWORK.state_dict(), main_model_path)
    logger.info(f"save {main_model_path.name}")
    torch.save(TARGET_NETWORK.state_dict(), target_model_path)
    logger.info(f"save {target_model_path.name}")


def main():
    executor = ThreadPoolExecutor(max_workers=tasks + 2)
    for i in range(tasks):
        executor.submit(play_game, i)

    executor.submit(batch_trainer, pack_main)
    executor.submit(batch_trainer, pack_target)

    start_time = datetime.now()
    while datetime.now() - start_time < cfg.TIME_LIMIT:
        time.sleep(1)  # 少し待ってから終了処理を行う

    stop_event.set()
    save_models()
    clear_queues()
    executor.shutdown()
    logger.info("All threads have been successfully terminated.")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        stop_event.set()
        raise e
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping threads...")
        stop_event.set()
        clear_queues()
        save_models()
