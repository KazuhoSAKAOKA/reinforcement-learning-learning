# ====================
# モンテカルロ木探索の作成
# ====================

# パッケージのインポート
from game_board import GameBoard, GameRelativeResult
from math import sqrt
import numpy as np

# パラメータの準備
PV_EVALUATE_COUNT = 50 # 1推論あたりのシミュレーション回数（本家は1600）
#C_PUCT = 2.5
C_BASE = 19652
C_INIT = 1.25
ALPHA = 0 # 0.3
EPSILON = 0 #0.25

class Parameter:
    def __init__(self, c_base : float, c_init : float, alpha : float, epsilon : float):
        self.c_base = c_base
        self.c_init = c_init
        self.alpha = alpha
        self.epsilon = epsilon

parameter = Parameter(C_BASE, C_INIT, ALPHA, EPSILON)

# ボルツマン分布
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

# ノードのリストを試行回数のリストに変換
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

# モンテカルロ木探索のスコアの取得
def pv_mcts_scores(board : GameBoard, temperature : float, evaluate_count : int, predict_alpha, predict_beta):
    # モンテカルロ木探索のノードの定義
    class Node:
        # ノードの初期化
        def __init__(self, board, p, predict_alpha, predict_beta, is_root = False):
            self.board = board # 状態
            self.p = p # 方策
            self.w = 0 # 累計価値
            self.n = 0 # 試行回数
            self.child_nodes = None  # 子ノード群
            self.predict_alpha = predict_alpha
            self.predict_beta = predict_beta
            self.is_root = is_root
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
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # ニューラルネットワークの推論で方策と価値を取得
                policies, value = predict_alpha(self.board)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                self.child_nodes = []
                for action, policy in zip(self.board.get_legal_actions(), policies):
                    succeed, next = self.board.transit_next(action)
                    if not succeed:
                        Exception("Invalid action")
                    self.child_nodes.append(Node(next, policy, predict_beta, predict_alpha))
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
            # アーク評価値の計算
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []

            if self.is_root and parameter.alpha > 0:
                noises = np.random.dirichlet([parameter.alpha] * len(self.child_nodes))

            #if self.board.get_turn() == 4:
            #    print("============================ begin arc value ============================ ")
            for i, child_node in enumerate(self.child_nodes):
                q_value = (-child_node.w / child_node.n if child_node.n else 0.0)
                #value2 = C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n)
                cs = np.log((1 + t + parameter.c_base) / parameter.c_base) + parameter.c_init
                p = child_node.p
                if self.is_root and parameter.alpha > 0:
                    p = (1 - parameter.epsilon) * p + parameter.epsilon * noises[i]
                u_value = cs * child_node.p * sqrt(t) / (1 + child_node.n)
                arc_value = q_value + u_value
                pucb_values.append(arc_value)
            #    if self.board.get_turn() == 4:
            #        print("========== arc value ==========")
            #        print(child_node.board)
            #        print("policy:{0}, v={1}, w={2}, n={3}, v1:{4},v2:{5}".format(child_node.p, arc_value, child_node.w, child_node.n, q_value, u_value))
            #        print("===============================")
            # アーク評価値が最大の子ノードを返す
            max_pucb_index = np.argmax(pucb_values)
            #if self.board.get_turn() == 4:
            #    print("============================ selected action:{0} ============================ ".format(max_pucb_index))
            return self.child_nodes[max_pucb_index]

    # 現在の局面のノードの作成
    root_node = Node(board, 0, predict_alpha, predict_beta, True)

    # 複数回の評価の実行
    for _ in range(evaluate_count):
        root_node.evaluate()

    # 合法手の確率分布
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0: # 最大値のみ1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else: # ボルツマン分布でバラつき付加
        scores = boltzman(scores, temperature)
    return scores

# モンテカルロ木探索で行動選択
def pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action



# 動作確認
if __name__ == '__main__':
    # モデルの読み込み
    path = sorted(Path('./model').glob('*.h5'))[-1]
    model = load_model(str(path))

    # 状態の生成
    state = State()

    # モンテカルロ木探索で行動取得を行う関数の生成
    next_action = pv_mcts_action(model, 1.0)
 
    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の取得
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

        # 文字列表示
        print(state)