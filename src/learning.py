import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from queue import Queue

import numpy as np
import torch
from torch import nn, optim

import config_2048 as cfg
from arg import args
from common import get_values, save_models, write_make_input
from config_2048 import MAIN_NETWORK, TARGET_NETWORK
from game_2048_3_3 import State

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
tasks = min(os.cpu_count() - 2, 6)
stop_event = threading.Event()  # スレッド終了用に使用
bat_size = 1024  # バッチサイズ
criterion = nn.MSELoss()  # 損失関数
optimizer_main = optim.Adam(MAIN_NETWORK.parameters(), lr=0.001)
optimizer_target = optim.Adam(TARGET_NETWORK.parameters(), lr=0.001)
pack_main = {
    "model": MAIN_NETWORK,
    "optimizer": optimizer_main,
    "name": "main",
    "queue": Queue(tasks * 2),
}
pack_target = {
    "model": TARGET_NETWORK,
    "optimizer": optimizer_target,
    "name": "target",
    "queue": Queue(tasks * 2),
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

        logger.debug(f"{bd_list=}\n\t{board_cp=}")
        for bd in bd_list:
            queue.put(
                {
                    "board": bd,
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
            # packs.reverse()
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
                            f"GAMEOVER: {thread_id=:02d} {count=:03d} {bd.score=:04d} {turn=:04d} {packs[0]['name'].ljust(7)}queue_size={packs[0]['queue'].qsize():04d} {init_eval_1=:.2f} {init_eval_2=:.2f}"
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
    # 重み更新の頻度
    update_target_every = args.target_update_freq
    while not stop_event.is_set():
        train_count += 1
        while len(records) != bat_size and not stop_event.is_set():
            records.append(pack["queue"].get())
        train(records, pack, train_count)

        if train_count % update_target_every == 0:
            TARGET_NETWORK.load_state_dict(MAIN_NETWORK.state_dict())
            logger.info(f"Update target network at {train_count=}, {pack['name']}")

        records.clear()
    return train_count


def clear_queues():
    while pack_main["queue"].qsize() > 0 and pack_target["queue"].qsize() > 0:
        pack_main["queue"].get()
        pack_target["queue"].get()
    logger.info("Queues cleared, stopping threads...")


def main():
    executor = ThreadPoolExecutor(max_workers=tasks + 1)
    for i in range(tasks):
        executor.submit(play_game, i)

    executor.submit(batch_trainer, pack_main)

    start_time = datetime.now()
    save_count = 0
    last_save_time = start_time
    save_interval = timedelta(hours=24)
    while datetime.now() - start_time < cfg.TIME_LIMIT and not stop_event.is_set():
        if cfg.TIME_LIMIT.total_seconds() >= save_interval.total_seconds():
            if datetime.now() - last_save_time >= save_interval:
                save_count += 1
                save_models(save_count)
                last_save_time = datetime.now()
        time.sleep(1)  # 少し待ってから終了処理を行う

    stop_event.set()
    save_models()
    clear_queues()
    logger.info("All threads have been successfully terminated.")
    return executor


if __name__ == "__main__":
    try:
        executor = main()
    except Exception as e:
        logger.exception(e)
        stop_event.set()
        raise e
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping threads...")
        stop_event.set()
        clear_queues()
        save_models()
    finally:
        if "executor" in locals():
            executor.shutdown(wait=True)
        logger.info("Program terminated gracefully.")
        if args.train_after_play:
            logger.info("Starting play after training...")
            from play import main as play_main

            play_main()
