import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import numpy as np

from common import calc_progress, get_values
from config_2048 import MAIN_NETWORK, TARGET_NETWORK, get_model_name
from game_2048_3_3 import State

stop_event = threading.Event()
logger = logging.getLogger(__name__)
games_played = 0
GAMES_TO_PLAY = 10
queue = Queue(GAMES_TO_PLAY)
tasks = os.cpu_count()

SAVE_DIR = Path("board_data") / get_model_name()
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def play_game(thread_id: int):
    global games_played
    packs = [{"model": MAIN_NETWORK}, {"model": TARGET_NETWORK}]
    while not stop_event.is_set():
        with threading.Lock():
            if games_played >= GAMES_TO_PLAY:
                stop_event.set()
                break
            games_played += 1

        states = []
        after_states = []
        evals = []
        bd = State()
        bd.initGame()
        turn = 0
        while not bd.isGameOver():
            turn += 1
            canmov = [bd.canMoveTo(i) for i in range(4)]
            copy_bd = bd.clone()
            main_values, target_values = get_values(canmov, copy_bd, packs)
            progress = calc_progress(bd.board.copy())
            # main_valuesから最大の評価値を持つインデックスを取得
            main_max_index = np.argmax(main_values)

            states.append((bd.board.copy(), progress))
            evals.append((main_values, progress))

            bd.play(main_max_index)
            after_states.append((bd.board.copy(), calc_progress(bd.board.copy())))
            bd.putNewTile()

        queue.put(
            {
                "thread_id": thread_id,
                "states": states,
                "after_states": after_states,
                "evals": evals,
                "gameover_turn": turn,
                "gameover_progress": calc_progress(bd.board),
                "gameover_score": bd.score,
            }
        )


def main():
    executor = ThreadPoolExecutor(max_workers=tasks)
    for i in range(tasks):
        executor.submit(play_game, i)

    data = []
    while len(data) < GAMES_TO_PLAY:
        data.append(queue.get())
        logger.info(f"Collected {len(data)}/{GAMES_TO_PLAY} games.")

    executor.shutdown(wait=True)
    stop_event.set()
    logger.info(f"Played {games_played} games.")

    scores = [game_data["gameover_score"] for game_data in data]
    try:
        with (
            open(SAVE_DIR / "state.txt", "w") as f_state,
            open(SAVE_DIR / "eval.txt", "w") as f_eval,
            open(SAVE_DIR / "after-state.txt", "w") as f_after,
        ):

            for i, game_data in enumerate(data):
                states = game_data["states"]
                after_states = game_data["after_states"]
                evals = game_data["evals"]
                gameover_info = f"gameover_turn: {game_data['gameover_turn']}; game: {i+1}; progress: {game_data['gameover_progress']}; score: {game_data['gameover_score']}"

                state_strs = []
                eval_strs = []
                after_state_strs = []
                for state, after_state, eval_data in zip(states, after_states, evals):
                    state_str = f"{' '.join(map(str, state[0]))} {state[1]}"
                    after_state_str = (
                        f"{' '.join(map(str, after_state[0]))} {after_state[1]}"
                    )
                    eval_str = f"{' '.join(str(float(ev)) for ev in eval_data[0])} {eval_data[1]}"
                    state_strs.append(state_str)
                    eval_strs.append(eval_str)
                    after_state_strs.append(after_state_str)
                # 1ゲーム分のデータを追記
                f_state.write("\n".join(state_strs) + f"\n{gameover_info}\n")
                f_eval.write("\n".join(eval_strs) + f"\n{gameover_info}\n")
                f_after.write("\n".join(after_state_strs) + f"\n{gameover_info}\n")

    except IOError as e:
        logger.error(f"File writing error: {e}")
        return

    # 最終的な統計情報を計算して表示
    if scores:
        logger.info(f"Average Score: {np.mean(scores):.2f}")
        logger.info(f"Median Score: {np.median(scores)}")
        logger.info(f"Max Score: {np.max(scores)}")
        logger.info(f"Min Score: {np.min(scores)}")

    logger.info("End Free Play")


if __name__ == "__main__":
    main()
