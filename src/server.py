# python_server.py
import json  # 配列のシリアライズ/デシリアライズにJSONを使用
import logging
import socket

import numpy as np
import torch

from common import write_make_input
from config_2048 import DEVICE, MAIN_NETWORK

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"  # localhost
PORT = 65432  # Arbitrary non-privileged port


def get_eval(board: np.ndarray, model: torch.nn.Module):
    x = torch.zeros(1, 99, device=DEVICE)
    write_make_input(board, x[0, :])
    model.eval()  # モデルを評価モードに設定
    eval = model.forward(x)
    return eval.item()


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    logger.info(f"Python server listening on {HOST}:{PORT}...")
    conn, addr = s.accept()
    with conn:
        logger.info(f"Connected by {addr}")
        while True:
            data = conn.recv(
                4096
            )  # 受信バッファを大きくする（JSON文字列が長くなる可能性を考慮）
            if not data:
                break

            try:
                # 受信したJSON文字列をPythonのリストにデコード
                decoded_data = data.decode("utf-8")
                board = json.loads(decoded_data)
                logger.info(f"Received board from C++: {board}")

                # 盤面を評価
                evaluation = get_eval(np.array(board, dtype="int64"), MAIN_NETWORK)
                logger.info(f"Evaluated value: {evaluation}")

                # 評価値をJSON形式でC++に返信
                response_message = json.dumps({"evaluation": evaluation})
                conn.sendall(response_message.encode("utf-8"))

            except json.JSONDecodeError:
                logger.info("Received invalid JSON data.")
                conn.sendall(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            except Exception as e:
                logger.info(f"An error occurred: {e}")
                conn.sendall(json.dumps({"error": str(e)}).encode("utf-8"))

logger.info("Python server disconnected.")
