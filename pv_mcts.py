# ====================
# モンテカルロ木探索の作成
# ====================

# パッケージのインポート
from typing import Callable, Tuple
from game_board import GameBoard, GameRelativeResult
from math import sqrt
import numpy as np
from parameter import PARAM

DEBUG_OUTPUT = False


# ボルツマン分布
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    total = sum(xs)
    if total == 0:
        return xs
    return [x / sum(xs) for x in xs]

# ノードのリストを試行回数のリストに変換
def nodes_to_scores(nodes):
    scores = [c.n for c in nodes]
    return np.array(scores)

# モンテカルロ木探索のノードの定義
class Node:
    # ノードの初期化
    def __init__(self
                    , board : GameBoard
                    , policy : float
                    , predict_alpha: Callable[[GameBoard], Tuple[np.ndarray, float]]
                    , predict_beta: Callable[[GameBoard], Tuple[np.ndarray, float]]
                    , is_root : bool = False):
        self.board = board # 状態
        self.policy :float= policy # 方策
        self.w :float= 0.0 # 累計価値
        self.n :int = 0 # 試行回数
        self.child_nodes = None  # 子ノード群
        self.predict_alpha = predict_alpha
        self.predict_beta = predict_beta
        self.is_root = is_root
    def __repr__(self) -> str:
        return 'Node(policy:{0},w:{1},n:{2})'.format(self.policy, self.w, self.n)
    def get_detail(self)->str:
        return 'Node(policy:{0},w:{1},n:{2} {3},{4},{5})'.format(self.policy, self.w, self.n, type(self.policy), type(self.w), type(self.n))
    # 局面の価値の計算
    def evaluate(self):
        # ゲーム終了時
        done, result = self.board.judge_last_action()
        if done:
            value = 0.0
            if result == GameRelativeResult.draw:
                pass
            elif result == GameRelativeResult.win_last_play_player:
                value -= 1.0
            else:
                value += 1.0
            # 累計価値と試行回数の更新
            self.w += value
            self.n += 1

            if DEBUG_OUTPUT:
                print("=== MCTS ===")
                print(self.board)
                print("policy:{0}, v={1}".format(0, value))
                print("============")
                print()
            
            return value

        # 子ノードが存在しない時
        if not self.child_nodes:
            # ニューラルネットワークの推論で方策と価値を取得
            policies, value = self.predict_alpha(self.board)

            if DEBUG_OUTPUT:
                print("=== MCTS ===")
                print(self.board)
                print("policy:{0}, v={1}".format(policies, value))
                print("policy type={}".format(policies.dtype))
                print("============")
                print()
            
            # 累計価値と試行回数の更新
            self.w += value
            self.n += 1

            # 子ノードの展開
            self.child_nodes = []
            for action in self.board.get_legal_actions():
                next, succeed = self.board.transit_next(action)
                if not succeed:
                    Exception("Invalid action")
                self.child_nodes.append(Node(next, policies[action], self.predict_beta, self.predict_alpha))
            return value

        # 子ノードが存在する時
        else:
            # アーク評価値が最大の子ノードの評価で価値を取得
            value = -self.next_child_node().evaluate()

            # 累計価値と試行回数の更新
            self.w += value
            self.n += 1
            return value

    # アーク評価値が最大の子ノードを取得
    def next_child_node(self):
        if self.is_root and PARAM.alpha > 0:
            noises = np.random.dirichlet([PARAM.alpha] * len(self.child_nodes))

        # 展開していない場合はポリシーをそのまま使用
        if self.n == 0:
            policies = [c.policy for c in self.child_nodes]
            if self.is_root and PARAM.alpha > 0:
                policies = [(1 - PARAM.epsilon) * p + PARAM.epsilon * noises[i] for i, p in enumerate(policies)]
            return self.child_nodes[np.argmax(policies)]
        
        # アーク評価値の計算
        t = sum(nodes_to_scores(self.child_nodes))
        pucb_values = []


        for i, child_node in enumerate(self.child_nodes):
            q_value = (-child_node.w / child_node.n if child_node.n else 0.0)
            cs = np.log((1 + t + PARAM.c_base) / PARAM.c_base) + PARAM.c_init
            child_policy = child_node.policy
            # ルートであればノイズを付加
            if self.is_root and PARAM.alpha > 0:
                child_policy = (1 - PARAM.epsilon) * child_policy + PARAM.epsilon * noises[i]
            u_value = cs * child_policy * sqrt(t) / (1 + child_node.n)
            arc_value = q_value + u_value
            pucb_values.append(arc_value)

        # アーク評価値が最大の子ノードを返す
        max_pucb_index = np.argmax(pucb_values)
        return self.child_nodes[max_pucb_index]

def pv_mcts_core(board : GameBoard
                   , evaluate_count : int
                   , predict_alpha :Callable[[GameBoard], Tuple[np.ndarray, float]]
                   , predict_beta:Callable[[GameBoard], Tuple[np.ndarray, float]])->Node:

    # 現在の局面のノードの作成
    root_node = Node(board, 0, predict_alpha, predict_beta, True)
    # ルートの子ノードを展開する
    root_node.evaluate()
    if len(root_node.child_nodes) > 1:
        # 複数回の評価の実行
        for _ in range(evaluate_count):
            root_node.evaluate()
    elif len(root_node.child_nodes) == 1:
        root_node.child_nodes[0].n = 1

    return root_node

# モンテカルロ木探索のスコアの取得
def pv_mcts_scores(board : GameBoard
                   , evaluate_count : int
                   , predict_alpha :Callable[[GameBoard], Tuple[np.ndarray, float]]
                   , predict_beta:Callable[[GameBoard], Tuple[np.ndarray, float]]):
    root_node = pv_mcts_core(board=board, evaluate_count=evaluate_count, predict_alpha=predict_alpha, predict_beta=predict_beta)
    scores = nodes_to_scores(root_node.child_nodes)

    with open('DEBUG_OUT.txt', 'a' ) as f:
        f.write("======= scores ========\n")
        for i, node in enumerate(root_node.child_nodes):
            f.write('{}:node{}\n'.format(i, node.get_detail()))
        f.write('board={}\n'.format(board))
        f.write('score={}\n'.format(scores))

    return scores

def pv_mcts_policies(board : GameBoard
                   , evaluate_count : int
                   , predict_alpha :Callable[[GameBoard], Tuple[np.ndarray, float]]
                   , predict_beta:Callable[[GameBoard], Tuple[np.ndarray, float]])->np.ndarray:
    root_node = pv_mcts_core(board=board, evaluate_count=evaluate_count, predict_alpha=predict_alpha, predict_beta=predict_beta)
    scores = nodes_to_scores(root_node.child_nodes)    
    # 行動空間に対する確率分布の取得　行動できないアクションは0
    policies = np.zeros(board.get_output_size(), dtype=np.float32)
    total = sum(scores)
    if total > 0:
        for s, c in zip(scores, root_node.child_nodes):
            policies[c.board.get_last_action()] = s / total
    else:
        legal_actions = board.get_legal_actions()
        for action in legal_actions:
            policies[action] = 1.0 / len(legal_actions)

    with open('DEBUG_OUT.txt', 'a' ) as f:
        f.write('policies={}\n'.format(policies))

    return policies



'''
def pv_mcts_policies_boltzman(board : GameBoard
                            , evaluate_count : int
                            , predict_alpha :Callable[[GameBoard], Tuple[np.ndarray, float]]
                            , predict_beta:Callable[[GameBoard], Tuple[np.ndarray, float]]):
    root_node = pv_mcts_core(board=board, evaluate_count=evaluate_count, predict_alpha=predict_alpha, predict_beta=predict_beta)
    scores = nodes_to_scores(root_node.child_nodes)

    with open('DEBUG_OUT.txt', 'a' ) as f:
        f.write("======= scores ========\n")
        for node in root_node.child_nodes:
            f.write('node{}\n'.format(node.get_detail()))

        f.write('board={}\n'.format(board))
        f.write('score={}\n'.format(scores))

        div = sum(scores) if sum(scores) else 1
        scores = np.array([x / div for x in scores])

        f.write('avereged,score={}\n'.format(scores))

        scores = boltzman(scores, PARAM.temperature)

        f.write('bolzman,score={}\n'.format(scores))


        # 行動空間に対する確率分布の取得　行動できないアクションは0
        ratio = np.zeros(board.get_output_size(), dtype=np.float32)
        total = sum(scores)
        if total > 0:
            for s, c in zip(scores, root_node.child_nodes):
                ratio[c.board.get_last_action()] = s / total
        else:
            legal_actions = board.get_legal_actions()
            for action in legal_actions:
                ratio[action] = 1.0 / len(legal_actions)

        f.write('ratio,ratio={}\n'.format(ratio))

    return ratio
'''
