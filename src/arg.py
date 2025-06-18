import argparse

arg_keys = []
parser = argparse.ArgumentParser(description="2048 DDQN")
game_conf_group = parser.add_argument_group("ゲーム設定")
game_conf_group.add_argument(
    "-m", "--model", type=str, help="モデルの選択", required=True
)
game_conf_group.add_argument("-s", "--seed", type=int, help="シード値")
game_conf_group.add_argument(
    "-r",
    "--restart",
    action="store_true",
    help="学習をリスタートする",
)
game_conf_group.add_argument(
    "-S",
    "--symmetry",
    action="store_true",
    help="対称性を考慮する",
)
game_conf_group.add_argument(
    "--sym_type",
    type=int,
    default=0,
    choices=[0, 1, 2, 3, 4],
    help="0: 通常, 1: ランダムで取得, 2: 重複削除, 3: 自分自身(index0)を選択, 4: 対称性index2を選択",
)
parser.add_argument(
    "-l",
    "--log",
    type=str,
    help="ログレベル",
    choices=["DEBUG", "INFO"],
    default="INFO",
)
parser.add_argument(
    "-H",
    "--hours",
    type=int,
    default=0,
    help="学習時間",
)
parser.add_argument(
    "-L",
    "--load_model",
    action="store_true",
    help="学習済みモデルをロードする",
)
parser.add_argument(
    "--ddqn",
    type=int,
    help="0: ゲームごとに使用するモデルを切り替える, 1: ランダムにモデルを切り替える, 2: グローバルで切り替える",
    default=0,
)
parser.add_argument(
    "--turget_update_freq",
    type=int,
    help="ターゲットネットワークの更新頻度",
    default=10,
)

args = parser.parse_args()  # 4. 引数を解析
for action in game_conf_group._group_actions:
    # もしデフォルトと一致するのであれば
    if action.default is not None and getattr(args, action.dest) == action.default:
        continue
    arg_keys.append(action.dest)
game_conf = {k: getattr(args, k) for k in arg_keys}
