# ====================
# モンテカルロ木探索の作成
# ====================

# パッケージのインポート
from typing import Callable, Tuple
from game_board import GameBoard, GameRelativeResult
from math import sqrt
import numpy as np
from parameter import BrainParameter, NetworkType,ActionSelectorType, ExplorationParameter
from predictor import Prediction, DualNetworkPrediction, PolicyNetworkPrediction, Predictor
from montecarlo import playout
from logging import getLogger, DEBUG
logger = getLogger(__name__)

# ボルツマン分布
def boltzmann(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    total = sum(xs)
    if total == 0:
        return xs
    return [x / sum(xs) for x in xs]

class AbstractMctsNode:
    def __init__(self):
        self.w :float = 0.0 # 累計価値
        self.n :int = 0 # 試行回数
    def get_selected_count(self)->int:
        return self.n
    def get_value(self)->float:
        return self.w
    #abstractmethod
    def get_policy(self)->float:
        pass
    #abstractmethod
    def evaluate(self)->float:
        pass
    def __repr__(self)->str:
        return 'Node(w:{0},n:{1})'.format(self.w, self.n)

# ノードのリストを試行回数のリストに変換
def nodes_to_scores(nodes : list[AbstractMctsNode])->np.ndarray:
    scores = [c.get_selected_count() for c in nodes]
    return np.array(scores)



# MCTSearchで次のノードを選ぶ。
class NextNodeSelector:
    #abstractmethod
    def select(self, nodes : list[AbstractMctsNode])->AbstractMctsNode:
        pass

# UCB1により次のノードを選ぶ
class UCB1NextNodeSelector(NextNodeSelector):
    def select(self, nodes : list[AbstractMctsNode])->AbstractMctsNode:
        for node in nodes:
            if node.get_selected_count() == 0:
                return node

        scores = nodes_to_scores(nodes)
        t = np.sum(scores)
        ucb1_values = []
        for node in nodes:
            ucb1_values.append(-node.get_value() / node.get_selected_count() + sqrt(2 * np.log(t) / node.get_selected_count()))
        return nodes[np.random.choice(np.where(ucb1_values == np.max(ucb1_values))[0])]
    
# Policy-UCTにより次のノードを選ぶ
class PolicyUCTNextNodeSelector(NextNodeSelector):
    def __init__(self, c_base: float = 19652, c_init: float = 1.25):
        self.c_base = c_base
        self.c_init = c_init
    def select(self, nodes : list[AbstractMctsNode])->AbstractMctsNode:
        t = sum(nodes_to_scores(nodes))
        # 最初の一回はポリシーをそのまま使用
        if t == 0:
            policies = [c.get_policy() for c in nodes]
            return nodes[np.random.choice(np.where(policies == np.max(policies))[0])]
        
        # アーク評価値の計算
        pucb_values = []

        for i, node in enumerate(nodes):
            n = node.get_selected_count()
            q_value = (-node.get_value() / n  if n else 0.0)
            cs = np.log((1 + t + self.c_base) / self.c_base) + self.c_init
            policy = node.get_policy()
            u_value = cs * policy * sqrt(t) / (1 + n)
            arc_value = q_value + u_value
            pucb_values.append(arc_value)
        # アーク評価値が最大の子ノードを返す
        return nodes[np.random.choice(np.where(pucb_values == np.max(pucb_values))[0])]

# Policy-UCTにより次のノードを選ぶ+ディリクレノイズ
class WithDirichletPolicyUCTNextNodeSelector(PolicyUCTNextNodeSelector):

    def __init__(self, c_base: float = 19652, c_init: float = 1.25, alpha: float = 0.3, epsilon: float = 0.25):
        super().__init__(c_base, c_init)
        self.alpha = alpha
        self.epsilon = epsilon
    def select(self, nodes : list[AbstractMctsNode])->AbstractMctsNode:
        noises = np.random.dirichlet([self.alpha] * len(nodes))
        t = sum(nodes_to_scores(nodes))
        # 最初の一回はポリシーをそのまま使用
        if t == 0:
            policies = [c.get_policy() for c in nodes]
            policies = [(1 - self.epsilon) * p + self.epsilon * noises[i] for i, p in enumerate(policies)]
            return nodes[np.random.choice(np.where(policies == np.max(policies))[0])]
        
        # アーク評価値の計算
        pucb_values = []

        for i, node in enumerate(nodes):
            n = node.get_selected_count()
            q_value = (-node.get_value() / n  if n else 0.0)
            cs = np.log((1 + t + self.c_base) / self.c_base) + self.c_init
            policy = node.get_policy()
            policy = (1 - self.epsilon) * policy + self.epsilon * noises[i]
            u_value = cs * policy * sqrt(t) / (1 + n)
            arc_value = q_value + u_value
            pucb_values.append(arc_value)
        # アーク評価値が最大の子ノードを返す
        return nodes[np.random.choice(np.where(pucb_values == np.max(pucb_values))[0])]
    def get_child_node_selector(self)->'NextNodeSelector':
        return PolicyUCTNextNodeSelector(self.c_base, self.c_init)

# モンテカルロ木探索ノードの定義
class MctsNode(AbstractMctsNode):
    # ノードの初期化
    def __init__(self,
                game_board : GameBoard,
                is_root : bool,
                node_selector: NextNodeSelector):
        super().__init__()
        self.game_board = game_board
        self.child_nodes = None  # 子ノード群
        self.is_root = is_root
        self.node_selector = node_selector
    def __repr__(self) -> str:
        return 'Node(last_action={0},w:{1},n:{2})'.format(self.game_board.get_last_action(), self.w, self.n)
    
    #abstractmethod
    def evaluate_self(self)->float:
        pass
    #abstractmethod
    def is_expandable(self)->bool:
        pass
    #abstractmethod
    def expand(self):
        pass
    def get_last_action(self)->int:
        return self.game_board.get_last_action()
    
    # 局面の価値の計算
    def evaluate(self)->float:
        # ゲーム終了時
        done, result = self.game_board.judge_last_action()
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
            if logger.isEnabledFor(DEBUG):
                logger.debug("=== MCTS ===")
                logger.debug(self.game_board)
                logger.debug("policy:{0}, v={1}".format(0, value))
                logger.debug("============")
            
            return value

        # 子ノードが存在しない時
        if not self.child_nodes:
            value = self.evaluate_self()

            self.w += value
            self.n += 1
     
            if self.is_expandable():
                self.expand()
            return value

        # 子ノードが存在する時
        else:
            selected_node = self.node_selector.select(self.child_nodes)
            value = -selected_node.evaluate()

            # 累計価値と試行回数の更新
            self.w += value
            self.n += 1

            if logger.isEnabledFor(DEBUG):
                logger.debug('selected node:{0}, value:{1}, total:{2}, selected_cnt:{3}'.format(selected_node, value, self.w, self.n))

            return value

class RandomMctsNode(MctsNode):
    def __init__(self,
                game_board : GameBoard,
                is_root : bool,
                expand_limit : int,
                node_selector : NextNodeSelector = None):
        if node_selector is None:
            node_selector = UCB1NextNodeSelector()
        super().__init__(game_board=game_board, is_root=is_root, node_selector=node_selector)
        self.expand_limit = expand_limit
    def evaluate_self(self)->float:
        value = playout(game_board=self.game_board)
        return value

    def is_expandable(self)->bool:
        return self.expand_limit <= self.n

    def expand(self):
        actions = self.game_board.get_legal_actions()
        self.child_nodes = []
        for action in actions:
            next_board, succeed = self.game_board.transit_next(action)
            if succeed:
                self.child_nodes.append(RandomMctsNode(
                    game_board=next_board, 
                    is_root= False,
                    expand_limit=self.expand_limit, 
                    node_selector=self.node_selector))

class PolicyValueNetworkMctsNode(MctsNode):
    # ノードの初期化
    def __init__(self,
                game_board : GameBoard,
                is_root : bool,
                predictor_alpha: Predictor,
                predictor_beta: Predictor,
                policy : float,
                node_selector : NextNodeSelector = None,
                child_node_selector : NextNodeSelector = None):
        if(node_selector is None):
            node_selector = PolicyUCTNextNodeSelector()
        if child_node_selector is None and is_root:
            child_node_selector = PolicyUCTNextNodeSelector()
        super().__init__(
            game_board=game_board, 
            is_root=is_root, 
            node_selector=node_selector)
        self.child_node_selector = child_node_selector
        self.predictor_alpha = predictor_alpha
        self.predictor_beta = predictor_beta
        self.policy :float= policy # 方策
        self.policies = None
    def is_expandable(self)->bool:
        return True
    
    def get_policy(self)->float:
        return self.policy    
    def evaluate_self(self)->float:
        # ニューラルネットワークの推論で方策と価値を取得
        prediction = self.predictor_alpha(self.game_board)
        self.policies = prediction.policies
        return prediction.value

    def expand(self):
        if(self.policies is None):
            self.evaluate_self()

        # 子ノードの展開
        self.child_nodes = []
        for action in self.game_board.get_legal_actions():
            next, succeed = self.game_board.transit_next(action)
            if not succeed:
                Exception("Invalid action")
            self.child_nodes.append(PolicyValueNetworkMctsNode(
                    game_board=next,
                    is_root= False,
                    predictor_alpha= self.predictor_beta,
                    predictor_beta= self.predictor_alpha,
                    policy=self.policies[action],
                    node_selector=self.child_node_selector,
                    child_node_selector=self.child_node_selector))


class ActionSelector:
    #abstract method
    def select_action(self, root_node:MctsNode)->int:
        pass
class MaxActionSelector(ActionSelector):
    def select_action(self, root_node:MctsNode)->int:
        scores = nodes_to_scores(root_node.child_nodes)
        node_index = np.random.choice(np.where(scores == np.max(scores))[0])
        return root_node.child_nodes[node_index].get_last_action()

class RandomChoiceActionSelector(ActionSelector):
    def select_action(self, root_node:MctsNode)->int:
        scores = nodes_to_scores(root_node.child_nodes)
        node_index = np.random.choice(len(scores), p=scores)
        return root_node.child_nodes[node_index].get_last_action()

class BoltzmanActionSelector(ActionSelector):
    def __init__(self, exploration_param: ExplorationParameter):
        self.temperature = exploration_param.temperature
    def select_action(self, root_node:MctsNode)->int:
        scores = nodes_to_scores(root_node.child_nodes)
        node_index = np.random.choice(len(scores), p=boltzmann(scores, self.temperature))
        return root_node.child_nodes[node_index].get_last_action()

def nodes_to_ratios(activation_space: int, nodes : list[AbstractMctsNode])->list[float]:
    ratios = [0.0] * activation_space
    if len(nodes) == 1:
        ratios[nodes[0].get_last_action()] = 1.0
        return ratios
    t = sum(nodes_to_scores(nodes))
    for node in nodes:
        ratios[node.get_last_action()] = float(node.get_selected_count()) / float(t)
    return ratios

class ActionSelectorFactory:
    @staticmethod
    def create_action_selector(action_selector_type: ActionSelectorType, exploration_param: ExplorationParameter)->ActionSelector:
        if action_selector_type == ActionSelectorType.max:
            return MaxActionSelector()
        elif action_selector_type == ActionSelectorType.random:
            return RandomChoiceActionSelector()
        elif action_selector_type == ActionSelectorType.boltzmann:
            return BoltzmanActionSelector(exploration_param=exploration_param)
        raise Exception('Unknown type')

class MonteCarloTreeSearcher:
    def __init__(self, brain_param: BrainParameter):
        self.evaluate_count = brain_param.mcts_evaluate_count
        self.action_selector = ActionSelectorFactory.create_action_selector(brain_param.action_selector_type, exploration_param=ExplorationParameter())
    def __call__(self, root_node:MctsNode)->int:
        return self.execute(root_node=root_node)
    def execute(self, root_node:MctsNode)->int:
        
        # ルートの子ノードを展開する
        while not root_node.is_expandable():
            root_node.evaluate()
        root_node.expand()

        if len(root_node.child_nodes) == 1:
            return root_node.child_nodes[0].get_last_action()

        # 評価
        for _ in range(self.evaluate_count):
            root_node.evaluate()
        return self.action_selector.select_action(root_node)
    def get_action_rations(self, root_node:MctsNode)->list[float]:
        return nodes_to_ratios(activation_space=root_node.game_board.get_output_size(),nodes=root_node.child_nodes)
    


