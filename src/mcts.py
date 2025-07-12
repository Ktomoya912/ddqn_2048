import logging
import math
import random
import sys
from typing import Any

import numpy as np  # numpyを直接使用
import torch

from common import write_make_input
from config_2048 import DEVICE, MAIN_NETWORK
from game_2048_3_3 import State

logger = logging.getLogger(__name__)
MAIN_NETWORK.eval()


def hash_board(state: State) -> int:
    """
    ボードの状態をハッシュ値に変換する関数。
    これはMCTSの状態ノードのハッシュキーとして使用されます。

    Args:
        board (np.ndarray): ゲームボードの状態（numpy配列）。

    Returns:
        int: ハッシュ値。
    """
    hash_value = 100
    base = 12
    for i in range(9):
        hash_value = hash_value * base + state.board[i]
    return hash_value


def calc_ev(board: np.ndarray):
    """
    ニューラルネットワークを用いてボードを評価する関数。
    これはMCTSの評価関数として使用されます。

    Args:
        board (np.ndarray): ゲームボードの状態（numpy配列）。

    Returns:
        float: ニューラルネットワークによるボードの評価値。
    """
    inputs = torch.zeros(1, 99, device="cpu")
    write_make_input(board, inputs[0, :])
    inputs = inputs.to(DEVICE)
    result = MAIN_NETWORK.forward(inputs)
    logger.debug(f"calc_ev: {result.data[0]}")
    return float(result.data[0])


class Edge:
    """MCTSの辺（アクション）を表すクラス。"""

    def __init__(self, tag, reward, child_node):
        self.tag = tag  # アクション（例：0:上, 1:下, 2:左, 3:右、またはタイル配置位置）
        self.r = reward  # このアクションから得られる即時報酬（状態遷移のスコア差など）
        self.cnode = child_node  # 子ノード（次の状態）
        self.count = 0  # この辺が選択された回数

    def __str__(self):
        return f"({self.tag}, {self.r}, {hex(id(self.cnode))})"


class Node:
    """MCTSのノード（状態）を表すクラス。"""

    def __init__(self, state: State, ev: float):
        self.state = state  # ゲームの状態 (Stateオブジェクト)
        self.visit_count = 0  # このノードが訪問された回数
        self.q_value = 0.0  # このノードの評価値（アクション選択の基準）
        self.expand_flg = False  # このノードが展開されたかを示すフラグ
        self.children: list[Edge] = []  # このノードからの辺（Edgeオブジェクトのリスト）
        self.ev = ev  # 評価関数（NN）による評価値

    def get_q_value(self):
        return self.q_value

    def __str__(self):
        children_str = "".join(str(e) for e in self.children)
        return (
            f"node p={hex(id(self))} vc={self.visit_count} Qv={self.q_value:.3f} "
            f"ef={int(self.expand_flg)} cs={len(self.children)} ev={self.ev:.3f} "
            f"children: {children_str}"
        )


class MCTSSearcher:
    """モンテカルロ木探索を実行するクラス。"""

    def __init__(
        self,
        simulations: int = 1000,
        random_turn: int = 5,
        expand_count: int = 2,
        c_param: float = 1.414,
        boltzmann: bool = False,
        expectimax: bool = False,
    ):
        self.simulations = simulations
        self.random_turn = random_turn
        self.expand_count = expand_count
        self.c = c_param  # UCB計算の探索パラメータ
        self.boltzmann = boltzmann  # ボルツマン選択を使用するか
        self.expectimax = expectimax  # Expectimaxバックアップを使用するか

        # Stateオブジェクトはデフォルトでハッシュ可能ではないため、
        # ボードとスコアのタプルをハッシュキーとして使用します。
        # 状態ノードを格納するマップ (hashable_state -> Node)
        self.state_node_map: dict[int, Node] = {}
        # アフターステートノードを格納するマップ (hashable_state -> Node)
        self.afterstate_node_map: dict[int, Node] = {}

        self.number_ev_calc = 0  # 評価関数呼び出し回数カウンタ

    def find_or_create_node_state(self, state: State, nodemap: dict):
        """
        ステートに対応するノードを見つけるか、新しく作成する。
        """
        hvalue = hash_board(state)
        if hvalue not in nodemap:
            # 状態ノードのev_valueは最初は0で良い（アフターステートノードのみNN評価値を持つ）
            node = Node(state, 0.0)
            nodemap[hvalue] = node
        return nodemap[hvalue]

    def find_or_create_node_afterstate(self, afterstate: State, nodemap: dict):
        """
        アフターステートに対応するノードを見つけるか、新しく作成する。
        アフターステートノードはNNによる評価値を持つ。
        """
        hvalue = hash_board(afterstate)
        if hvalue not in nodemap:
            # アフターステートノードはNNによる評価値を持つ
            ev = calc_ev(afterstate.board)
            self.number_ev_calc += 1
            node = Node(afterstate, ev)
            nodemap[hvalue] = node
        return nodemap[hvalue]

    def expand_state(self, node: Node):
        """
        状態ノードを展開し、可能なアクション（移動方向）に対応するアフターステートノードを生成する。
        """
        if node.expand_flg:
            return
        node.expand_flg = True

        original_score = node.state.score

        for d in range(4):  # 0:上, 1:右, 2:下, 3:左
            temp_state = node.state.clone()  # 現在の状態をクローン

            if temp_state.canMoveTo(d):  # 有効な移動の場合
                temp_state.play(d)  # 移動を実行
                move_score_diff = temp_state.score - original_score

                # アフターステートノードを見つけるか作成
                cnode = self.find_or_create_node_afterstate(
                    temp_state, self.afterstate_node_map
                )

                # 辺を追加
                node.children.append(Edge(d, move_score_diff, cnode))

        logger.debug(f"state expanded {node}")

    def expand_afterstate(self, node: Node):
        """
        アフターステートノードを展開し、ランダムなタイル配置に対応する状態ノードを生成する。
        """
        if node.expand_flg:
            return
        node.expand_flg = True

        for i in range(9):  # 各空きセルについて
            if node.state.board[i] != 0:
                continue
            # 新しいタイル (2または4) を配置

            # 配置するタイルが1 (値2) の場合
            temp_state_1 = node.state.clone()
            temp_state_1.board[i] = 1
            new_state_1 = temp_state_1  # Stateオブジェクトを直接使用
            cnode_1 = self.find_or_create_node_state(new_state_1, self.state_node_map)
            node.children.append(Edge(i, 0, cnode_1))  # tagは配置位置

            # 配置するタイルが2 (値4) の場合
            temp_state_2 = node.state.clone()
            temp_state_2.board[i] = 2
            new_state_2 = temp_state_2  # Stateオブジェクトを直接使用
            cnode_2 = self.find_or_create_node_state(new_state_2, self.state_node_map)
            node.children.append(
                Edge(i + 10, 0, cnode_2)
            )  # tagは配置位置+10で区別（元のC++コードを踏襲）

        logger.debug(f"afterstate expanded {node}")

    def do_state_select(self, node: Node):
        """
        状態ノードからUCBまたはボルツマン選択に基づいて子ノード（辺）を選択する。
        """
        # 未訪問のノードがあれば優先して選択
        unvisited_edges = [e for e in node.children if e.count == 0]
        if unvisited_edges:
            # 未訪問のノードのうち、NN評価値が最も高いものを選択する
            # C++コードの100000.0は、未訪問ノードを非常に有利にするための大きな値
            retval = max(unvisited_edges, key=lambda e: e.cnode.ev + 100000.0)
            logger.debug(f"do_state_select: select unvisited edge {retval}")
            return retval

        if self.boltzmann:
            t = 100  # FIXME: 温度パラメータ。調整が必要な場合があります。
            vals = []
            for e in node.children:
                # Q_valueが未訪問ノードの時には使えない可能性があるため、evも考慮する
                vals.append(math.exp((e.cnode.get_q_value() + e.cnode.ev) / t))
            logger.debug(
                f"do_state_select: Boltzmann vals = {[f'{d:.3f}' for d in vals]}"
            )

            sum_vals = sum(vals)
            if sum_vals == 0:  # 全てのvalsが0の場合の対策
                return random.choice(node.children)

            r = random.random()
            logger.debug(f"do_state_select: Boltzmann r = {r:.3f}")

            retval = None
            for i, val in enumerate(vals):
                if r < val / sum_vals:
                    retval = node.children[i]
                    break
                r -= val / sum_vals

            if retval is None and node.children:  # 丸め誤差などによるフォールバック
                retval = node.children[-1]

            assert retval, "Boltzmann selection failed to pick an edge."
            logger.debug(f"do_state_select: Boltzmann select edge {retval}")
            return retval
        else:
            # UCBに基づいて子を選ぶ
            total_visit_count = sum(e.count for e in node.children)

            if total_visit_count == 0:
                # このケースは未訪問エッジの処理でカバーされるはず
                if node.children:
                    return random.choice(node.children)
                else:
                    return None

            lc = self.c
            if lc == -1:  # C++コードの-1は特殊な意味を持つ
                lc = -sys.float_info.max  # DBL_MAX に相当
                for e in node.children:
                    lc = max(lc, e.cnode.get_q_value())
            lc = max(lc, 1.0)  # lcは少なくとも1

            max_ucb = -sys.float_info.max  # DBL_MAX に相当
            retval = None
            ucb_values = []

            for e in node.children:
                if e.count == 0:
                    continue

                ucb = e.cnode.get_q_value() + lc * math.sqrt(
                    2.0 * math.log(total_visit_count) / e.count
                )
                ucb_values.append(f"{ucb:.3f}")
                if max_ucb < ucb:
                    max_ucb = ucb
                    retval = e

            logger.debug(f"do_state_select: UCB ucb = {' '.join(ucb_values)}")
            assert retval, "UCB selection failed to pick an edge."
            logger.debug(f"do_state_select: UCB select edge {retval}")
            return retval

    def do_afterstate_select(self, node: Node):
        """
        アフターステートノードから重み付きランダムまたは一様ランダムで子ノードを選択する。
        """
        if not node.children:
            return None

        # childrenは(pos, tile=1)と(pos, tile=2)のペアで並んでいると仮定
        # これは expand_afterstate でそのように生成されるため

        # 90%の確率で2 (tile value 1)、10%の確率で4 (tile value 2) を生成する方に対応
        if random.random() < 0.9:
            num = 0  # 2 (value 1) に対応するインデックスオフセット
        else:
            num = 1  # 4 (value 2) に対応するインデックスオフセット

        # どの位置にタイルを置くかを選ぶ
        pos_options = len(node.children) // 2
        if pos_options == 0:  # 空きセルがない場合（ありえないはずだが念のため）
            return random.choice(node.children)

        pos_idx = random.randrange(pos_options)

        retval = node.children[pos_idx * 2 + num]

        logger.debug(
            f"do_afterstate_select: pos = {pos_idx} num={num} select edge {retval}"
        )
        return retval

    def do_playout(self, node: Node):
        """
        プレイアウト（シミュレーション）を実行し、結果の評価値を返す。
        """
        retval = 0.0
        if self.random_turn == 0:
            retval = node.ev  # ランダムターンが0ならNN評価値をそのまま使用
        else:
            current_state = node.state.clone()  # 現在の状態をクローン
            reach_game_over = False

            for rt in range(self.random_turn):
                # 新しいタイルを生成
                current_state.putNewTile()
                if current_state.isGameOver():
                    retval = current_state.score - node.state.score
                    reach_game_over = True
                    break

                # 有効な手が見つかるまでランダムに移動
                can_move_dirs = [d for d in range(4) if current_state.canMoveTo(d)]
                if not can_move_dirs:  # 動かせる手がない場合
                    if current_state.isGameOver():  # ゲームオーバーなら終了
                        retval = current_state.score - node.state.score
                        reach_game_over = True
                        break
                    # そうでなければ、タイルが置かれるのを待つ（次のステップへ）
                    break  # 動かせない場合はこのターンで終了し、評価に移る

                d = random.choice(can_move_dirs)
                current_state.play(d)
                # playoutでは、NN評価は最終的な状態に対して行われるため、
                # ここでのスコア変化は最終的な評価値に影響しない。
                # ただし、MCTSの報酬計算 (e.r) では重要。

            if not reach_game_over:
                retval = calc_ev(current_state.board)  # 最終的なボード状態をNNで評価
                self.number_ev_calc += 1

        logger.debug(f"do_playout: retval = {retval:.3f}")
        logger.debug(f"do_playout: node.q_value updated {node.q_value:.3f} -> ")

        # Q値の更新
        if node.visit_count > 0:  # 訪問回数が0でないことを確認
            node.q_value = (
                (node.q_value * (node.visit_count - 1)) + retval
            ) / node.visit_count
        else:  # 初めての訪問の場合
            node.q_value = retval

        logger.debug(f" {node.q_value:.3f}")
        assert not math.isnan(node.q_value), "Q_value became NaN in do_playout"
        return retval

    def do_state_update(self, node: Node, ev_from_child: float):
        """
        状態ノードのQ値を更新する。
        """
        logger.debug(f"do_state_update: node {node}")
        logger.debug(f"do_state_update: node.q_value updated {node.q_value:.3f} -> ")

        if self.expectimax:
            # Expectimaxでは子ノードの最大値を取る
            max_q = -sys.float_info.max
            found_valid_child = False
            for e in node.children:
                if e.count == 0:
                    continue
                v = e.cnode.get_q_value() + e.r
                max_q = max(max_q, v)
                found_valid_child = True

            if found_valid_child:
                node.q_value = max_q
            else:  # 全ての辺が未訪問か、有効な辺がない場合
                node.q_value = ev_from_child  # 新しく得られた評価値を使用

            assert not math.isnan(
                node.q_value
            ), "Q_value became NaN in do_state_update (expectimax)"
        else:
            # 単純平均
            if node.visit_count > 0:
                node.q_value = (
                    (node.q_value * (node.visit_count - 1)) + ev_from_child
                ) / node.visit_count
            else:  # 初めての訪問の場合
                node.q_value = ev_from_child
            assert not math.isnan(
                node.q_value
            ), "Q_value became NaN in do_state_update (simple average)"

        logger.debug(f" {node.q_value:.3f}")

    def do_afterstate_update(self, node: Node, ev_from_child: float):
        """
        アフターステートノードのQ値を更新する。
        """
        logger.debug(f"do_afterstate_update: node {node}")
        logger.debug(
            f"do_afterstate_update: node.q_value updated {node.q_value:.3f} -> "
        )

        if self.expectimax:
            # Expectimaxでは子ノードの重み付き平均を取る
            total_sum = 0.0
            total_count = 0
            found_valid_child = False
            for i, e in enumerate(node.children):
                if e.count == 0:
                    continue
                # C++の元のコードでは(i % 2 == 0) ? 9 : 1 という重み付けがある
                # これはタイルが2(値1)と4(値2)の確率9:1に対応している可能性
                weight = 9 if (i % 2 == 0) else 1
                total_sum += e.cnode.get_q_value() * weight
                total_count += weight
                found_valid_child = True

            if total_count > 0 and found_valid_child:
                node.q_value = total_sum / total_count
            else:
                node.q_value = ev_from_child  # 全ての辺が未訪問の場合のフォールバック

            assert not math.isnan(
                node.q_value
            ), "Q_value became NaN in do_afterstate_update (expectimax)"
        else:
            # 単純平均
            if node.visit_count > 0:
                node.q_value = (
                    (node.q_value * (node.visit_count - 1)) + ev_from_child
                ) / node.visit_count
            else:
                node.q_value = ev_from_child
            assert not math.isnan(
                node.q_value
            ), "Q_value became NaN in do_afterstate_update (simple average)"

        logger.debug(f" {node.q_value:.3f}")

    def simulation_afterstate_rec(self, node: Node):
        """
        アフターステートからのシミュレーション（再帰）。
        """
        logger.debug(f"in simulation_afterstate_rec: {node}")
        node.visit_count += 1

        if node.visit_count > self.expand_count:
            self.expand_afterstate(node)

        if node.expand_flg:
            # 再帰的に木を下る
            e = self.do_afterstate_select(node)
            if e is None:
                return node.ev

            ev = self.simulation_state_rec(e.cnode) + e.r
            e.count += 1
            self.do_afterstate_update(node, ev)
            return ev
        else:
            # 展開されていない場合、プレイアウトを実行
            ev = self.do_playout(node)
            return ev

    def simulation_state_rec(self, node: Node):
        """
        状態からのシミュレーション（再帰）。
        """
        logger.debug(f"in simulation_state_rec: {node}")
        node.visit_count += 1

        # 状態ノードでは評価値を計算できないため、必ず展開する
        self.expand_state(node)
        if not node.children:
            logger.debug("exiting: game over with value 0")
            return 0.0

        # 再帰的に木を下る
        e = self.do_state_select(node)
        if e is None:
            logger.debug("exiting: no valid state move")
            return 0.0

        ev = self.simulation_afterstate_rec(e.cnode) + e.r
        e.count += 1
        self.do_state_update(node, ev)
        return ev

    def print_tree(self, node: Node, depth: int):
        """MCTSツリーをデバッグ用に表示する。"""
        if node.visit_count == 0:
            return
        logger.debug("  " * depth + str(node))
        for e in node.children:
            self.print_tree(e.cnode, depth + 1)

    def do_simulation(self, root: Node):
        """単一のシミュレーションを実行する。"""
        self.simulation_state_rec(root)

    def clear_cache(self):
        """キャッシュされたノードと辺をクリアする。"""
        # Pythonではガベージコレクションがあるため、明示的なdeleteは不要ですが、
        # 循環参照を避けるためにchildrenの参照をクリアし、マップもクリアする。
        for kv in self.state_node_map.values():
            kv.children.clear()
        self.state_node_map.clear()

        for kv in self.afterstate_node_map.values():
            kv.children.clear()
        self.afterstate_node_map.clear()

    def search(self, initial_state: State):
        """
        MCTS探索を実行し、最適な手を見つける。
        Args:
            initial_state (State): 探索を開始するゲームの状態オブジェクト。

        Returns:
            tuple: (最適な手, 各手の評価値のリスト)
                   最適な手は0-3（方向）、または-1（見つからない場合）。
                   評価値のリストは、[上, 下, 左, 右, 最大評価値] の形式。
        """
        self.number_ev_calc = 0
        root = self.find_or_create_node_state(initial_state, self.state_node_map)
        self.expand_state(root)

        logger.debug("starting search for ")
        logger.debug(initial_state.print())  # Stateクラスのprintメソッドを呼び出す

        for nsim in range(self.simulations):
            if self.number_ev_calc >= self.simulations and nsim >= self.simulations:
                break

            logger.debug(f"simulation {nsim} number_ev_calc {self.number_ev_calc}")
            logger.debug(self.print_tree(root, 0))
            logger.debug("starting simulation")
            self.do_simulation(root)

        max_v = -sys.float_info.max
        move = -1
        evals = [-sys.float_info.max] * 5  # [上, 下, 左, 右, 最大評価値]

        # 各子ノードの評価値を取得し、evalsに格納
        for e in root.children:
            v = e.cnode.get_q_value() + e.r
            if 0 <= e.tag < 4:  # アクションが方向（0-3）の場合
                evals[e.tag] = v

            if max_v < v:
                max_v = v
                move = e.tag  # 最大評価値の手を更新

        evals[4] = max_v
        return move, evals


class MCTSPlayer:
    """MCTSプレイヤー"""

    def __init__(self, simulations: int = 1000, **kwargs):
        self.simulations = simulations
        self.searcher_kwargs = kwargs

    def get_move(self, state: State) -> int:
        """最適な手を取得"""
        searcher = MCTSSearcher(simulations=self.simulations, **self.searcher_kwargs)
        move, _ = searcher.search(state)
        logger.info(f"Selected move: {move} with simulations {searcher.simulations}")
        return move

    def play_game(self, verbose: bool = False) -> dict[str, Any]:
        """ゲームを一回プレイ"""
        state = State()
        state.initGame()

        moves = []
        scores = []
        turn = 0

        while not state.isGameOver():
            if verbose:
                print(f"\nTurn {turn}")
                state.print()

            move = self.get_move(state)
            moves.append(move)

            if state.canMoveTo(move):
                state.play(move)
                state.putNewTile()
                scores.append(state.score)
                turn += 1
            else:
                break

        if verbose:
            print(f"\nGame Over! Final score: {state.score}")
            state.print()

        return {"score": state.score, "moves": moves, "scores": scores, "turns": turn}


# --- 使用例 ---
if __name__ == "__main__":
    # MCTSパラメータの設定
    simulations = 1000  # シミュレーション回数
    random_turn = 10  # プレイアウトのランダムターン数
    expand_count = 5  # ノード展開の閾値
    c_param = 1.0  # UCBのC値
    boltzmann = False  # ボルツマン選択を使用するか
    expectimax = False  # Expectimaxバックアップを使用するか

    # MCTSプレイヤーを作成
    player = MCTSPlayer(simulations=100)

    # 複数回実行して統計を取る
    results = []
    for i in range(10):
        result = player.play_game()
        results.append(result["score"])
        print(f"Game {i+1}: Score {result['score']}")

    print(f"\nAverage Score: {np.mean(results):.2f}")
    print(f"Max Score: {max(results)}")
    print(f"Min Score: {min(results)}")
    print(f"\nAverage Score: {np.mean(results):.2f}")
    print(f"Max Score: {max(results)}")
    print(f"Min Score: {min(results)}")
    print(f"Min Score: {min(results)}")
