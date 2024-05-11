from typing import Callable, Tuple
import numpy as np
from tensorflow.keras.models import Model
from brains import Brain
from game_board import GameBoard
from pv_mcts import pv_mcts_policies,boltzman
from selfplay_brain import SelfplayBrain, HistoryUpdater, HistoryUpdaterFactory, HistoryUpdaterType
from threadsafe_dict import ThreadSafeDict
from parameter import PARAM

def predict_core(model : Model, board : GameBoard)->Tuple[np.ndarray, float]:
    # 推論のための入力データのシェイプの変換
    x = board.reshape_to_input()
    #print(x)
    # 推論
    y = model.predict(x, batch_size=1, verbose=0)

    with open('DEBUG_OUT.txt', 'a' ) as f:
        f.write("======= PREDICT ========\n")
        f.write('board={}\n'.format(board))
        f.write('p,v={}\n'.format(y))
        f.write('ptype={}\n'.format(y[0].dtype))
        f.write('vtype={}\n'.format(y[1].dtype))

    # 方策の取得
        policies = y[0][0][:]
        policies /= np.sum(policies) if np.sum(policies) else 1 # 合計1の確率分布に変換
        f.write('policies={}\n'.format(policies.dtype))


        # 価値の取得
        value = y[1][0][0]
    return policies, value

# 推論
def predict(model : Model, board : GameBoard, ts_dict : ThreadSafeDict)->Tuple[np.ndarray, float]:
    if ts_dict is None:
        return predict_core(model, board)
    return ts_dict.get_or_add(board.to_state_key(), lambda: predict_core(model, board))


class NetworkMonteCarloTreeSearcher:
    def __init__(self, evaluate_count : int, model: Model, ts_dict : ThreadSafeDict):
        self.evaluate_count = evaluate_count
        self.model = model
        self.ts_dict = ts_dict
    def __call__(self, game_board: GameBoard)->np.ndarray:
        policies = pv_mcts_policies(game_board, self.evaluate_count, lambda x: predict(self.model, x, self.ts_dict) , lambda x: predict(self.model, x, self.ts_dict))
        return policies
    def __str__(self) -> str:
        return 'MCTS (evaluate_count={})'.format(self.evaluate_count)

class DualModelNetworkMonteCarloTreeSearcher(NetworkMonteCarloTreeSearcher):
    def __init__(self, evaluate_count : int, first_model: Model, second_model: Model, ts_dict : ThreadSafeDict):
        super().__init__(evaluate_count, None, ts_dict)
        self.first_model = first_model
        self.second_model = second_model
    def __call__(self, game_board: GameBoard)->np.ndarray:
        if game_board.is_first_player_turn():
            policies = pv_mcts_policies(game_board, self.evaluate_count, lambda x: predict(self.first_model, x, self.ts_dict) , lambda x: predict(self.second_model, x, self.ts_dict))
        else:
            policies = pv_mcts_policies(game_board, self.evaluate_count, lambda x: predict(self.second_model, x, self.ts_dict) , lambda x: predict(self.first_model, x, self.ts_dict))
        return policies
    def __str__(self) -> str:
        return 'DualModel MCTS (evaluate_count={})'.format(self.evaluate_count)

class PolicySelector:
    def __call__(self, policies:np.ndarray)->int:
        return np.argmax(policies)
    def __str__(self) -> str:
        return 'max selector'

class SelfplayPolicySelector:
    def __call__(self, policies:np.ndarray)->int:
        return np.random.choice(a=len(policies), p=policies)
    def __str__(self) -> str:
        return 'random choice selector'

class BoltmanPolicySelector:
    def __call__(self, policies:np.ndarray)->int:
        policies = boltzman(policies, PARAM.temperature)
        return np.random.choice(a=len(policies), p=policies)
    def __str__(self) -> str:
        return 'random choice selector(boltzman)'
    
class NetworkBrain(Brain):
    def __init__(self, network_mcts: NetworkMonteCarloTreeSearcher, policy_selector : PolicySelector):
        super().__init__()
        self.network_mcts = network_mcts
        self.policy_selector = policy_selector
        self.last_policies = None
        self.last_action = None
    def get_name(self):
        return 'NetworkBrain MCTS={0} selector={1}'.format(self.network_mcts, self.policy_selector)
    def select_action(self, game_board:GameBoard)->int:
        policies = self.network_mcts(game_board=game_board)
        self.last_policies = policies
        action = self.policy_selector(policies=policies)
        self.last_action = action
        return action
    def get_last_policies(self):
        return self.last_policies
    def get_last_action(self):
        return self.last_action

class SelfplayNetworkBrain(SelfplayBrain):
    def __init__(self, network_mcts: NetworkMonteCarloTreeSearcher, policy_selector : PolicySelector, history_updater:HistoryUpdater):
        super().__init__(history_updater=history_updater)
        self.network_mcts = network_mcts
        self.policy_selector = policy_selector
        self.last_policies = None
        self.last_action = None
    def get_name(self):
        return 'SelfplayNetworkBrain MCTS={0} selector={1}'.format(self.network_mcts, self.policy_selector)
    def select_action(self, game_board : GameBoard)->int:
        policies = self.network_mcts(game_board=game_board)
        self.last_policies = policies
        self.register_policies(game_board=game_board, policies=policies)
        action = self.policy_selector(policies=policies)
        self.last_action = action
        return action
    def get_last_policies(self):
        return self.last_policies
    def get_last_action(self):
        return self.last_action


class NetworkBrainFactory:
    @staticmethod
    def create_network_brain(evaluate_count : int, model: Model, ts_dict : ThreadSafeDict)->NetworkBrain:
        return NetworkBrain(
            network_mcts=NetworkMonteCarloTreeSearcher(evaluate_count=evaluate_count, model=model, ts_dict=ts_dict),
            policy_selector=PolicySelector())
    def create_dualmodel_network_brain(evaluate_count : int, first_model: Model, second_model: Model, ts_dict : ThreadSafeDict)->NetworkBrain:
        return NetworkBrain(
            network_mcts=DualModelNetworkMonteCarloTreeSearcher(evaluate_count=evaluate_count, first_model=first_model, second_model=second_model, ts_dict=ts_dict),
            policy_selector=PolicySelector())
    def create_selfplay_network_brain(evaluate_count : int, model: Model, ts_dict : ThreadSafeDict, history_updater:HistoryUpdater)->SelfplayNetworkBrain:
        return SelfplayNetworkBrain(
            network_mcts=NetworkMonteCarloTreeSearcher(evaluate_count=evaluate_count, model=model, ts_dict=ts_dict),
            policy_selector=SelfplayPolicySelector(),
            history_updater=history_updater)
    def create_selfplay_dualmodel_network_brain(evaluate_count : int, first_model: Model, second_model: Model, ts_dict : ThreadSafeDict, history_updater:HistoryUpdater)->SelfplayNetworkBrain:
        return SelfplayNetworkBrain(
            network_mcts=DualModelNetworkMonteCarloTreeSearcher(evaluate_count=evaluate_count, first_model=first_model, second_model=second_model, ts_dict=ts_dict),
            policy_selector=SelfplayPolicySelector(),
            history_updater=history_updater)

'''
class DualModelNetworkBrain(Brain):
    def __init__(self, evaluate_count : int, first_model : Model, second_model : Model, ts_dict : ThreadSafeDict):
        super().__init__()
        self.evaluate_count = evaluate_count
        self.first_model = first_model
        self.second_model = second_model
        self.last_policies = None
        self.ts_dict = ts_dict
    def get_name(self):
        return "DualModelNetworkBrain"
    
    def select_action(self, board)->int:
        if board.is_first_player_turn():
            ratios = pv_mcts_policies(board, self.evaluate_count, lambda x: predict(self.first_model, x, self.ts_dict) , lambda x: predict(self.second_model, x, self.ts_dict))
        else:
            ratios = pv_mcts_policies(board, self.evaluate_count, lambda x: predict(self.second_model, x, self.ts_dict) , lambda x: predict(self.first_model, x, self.ts_dict))
        self.last_policies = ratios
        action = np.argmax(ratios)
        return action
    def get_last_policies(self):
        return self.last_policies


class SelfplayNetworkBrain(SelfplayBrain):
    def __init__(self, evaluate_count : int, model: Model, ts_dict : ThreadSafeDict):
        super().__init__()
        self.evaluate_count = evaluate_count
        self.model = model
        self.ts_dict = ts_dict
    def get_name(self):
        return "SelfplayNetworkBrain"
    def select_action(self, board : GameBoard)->int:
        policies = pv_mcts_policies(board, self.evaluate_count, lambda x: predict(self.model, x, self.ts_dict) , lambda x: predict(self.model, x, self.ts_dict))
        action = np.random.choice(range(0, board.get_output_size()), p=policies)
        self.register_policies(board, policies)
        return action
    
class SelfplayDualModelNetworkBrain(SelfplayBrain):
    def __init__(self, evaluate_count : int, first_model : Model, second_model : Model, ts_dict : ThreadSafeDict):
        super().__init__()
        self.evaluate_count = evaluate_count
        self.first_model = first_model
        self.second_model = second_model
        self.last_policies = None
        self.ts_dict = ts_dict
    def get_name(self):
        return "SelfplayDualModelNetworkBrain"
    def select_action(self, board : GameBoard)->int:
        if board.is_first_player_turn():
            policies = pv_mcts_policies(board, self.evaluate_count, lambda x: predict(self.first_model, x, self.ts_dict) , lambda x: predict(self.second_model, x, self.ts_dict))
        else:
            policies = pv_mcts_policies(board, self.evaluate_count, lambda x: predict(self.second_model, x, self.ts_dict) , lambda x: predict(self.first_model, x, self.ts_dict))
        action = np.random.choice(range(0, board.get_output_size()), p=policies)
        self.register_policies(board, policies)
        return action

class SelfplayNetworkBrainWithBoltman(SelfplayBrain):
    def __init__(self, evaluate_count : int, model: Model, ts_dict : ThreadSafeDict):
        super().__init__()
        self.evaluate_count = evaluate_count
        self.model = model
        self.ts_dict = ts_dict
    def get_name(self):
        return "SelfplayNetworkBrainWithBoltman"
    def select_action(self, board : GameBoard)->int:
        policies = pv_mcts_policies_boltzman(board, self.evaluate_count, lambda x: predict(self.model, x, self.ts_dict) , lambda x: predict(self.model, x, self.ts_dict))
        action = np.random.choice(range(0, board.get_output_size()), p=policies)
        self.register_policies(board, policies)
        return action
    
class SelfplayDualModelNetworkBrainWithBoltman(SelfplayBrain):
    def __init__(self, evaluate_count : int, first_model : Model, second_model : Model, ts_dict : ThreadSafeDict):
        super().__init__()
        self.evaluate_count = evaluate_count
        self.first_model = first_model
        self.second_model = second_model
        self.last_policies = None
        self.ts_dict = ts_dict
    def get_name(self):
        return "SelfplayDualModelNetworkBrainWithBoltman"
    def select_action(self, board : GameBoard)->int:
        if board.is_first_player_turn():
            policies = pv_mcts_policies_boltzman(board, self.evaluate_count, lambda x: predict(self.first_model, x, self.ts_dict) , lambda x: predict(self.second_model, x, self.ts_dict))
        else:
            policies = pv_mcts_policies_boltzman(board, self.evaluate_count, lambda x: predict(self.second_model, x, self.ts_dict) , lambda x: predict(self.first_model, x, self.ts_dict))
        action = np.random.choice(range(0, board.get_output_size()), p=policies)
        self.register_policies(board, policies)
        return action
'''