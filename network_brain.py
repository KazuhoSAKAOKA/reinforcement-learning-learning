from typing import Callable, Tuple
import numpy as np
from tensorflow.keras.models import Model
from brains import Brain
from game_board import GameBoard
from self_play_brain import SelfplayBrain, HistoryUpdater,HistoryUpdaterFactory
from threadsafe_dict import ThreadSafeDict
from parameter import BrainParameter,ExplorationParameter
from predictor import Predictor
from mcts_node import MonteCarloTreeSearcher, PolicyValueNetworkMctsNode,RandomChoiceActionSelector,PolicyUCTNextNodeSelector,WithDirichletPolicyUCTNextNodeSelector
'''
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
'''
    
class NetworkBrain(Brain):
    def __init__(self, 
                predictor_first : Predictor,
                predictor_second : Predictor,
                brain_param:BrainParameter,
                exploration_param:ExplorationParameter):
        super().__init__()
        self.mcts = MonteCarloTreeSearcher(brain_param=brain_param)
        self.predictor_first = predictor_first
        self.predictor_second = predictor_second
        self.last_policies = None
        self.last_action = None
        self.node_selector = PolicyUCTNextNodeSelector(c_base=exploration_param.c_base, c_init=exploration_param.c_init)
    def get_name(self):
        return 'NetworkBrain ' # todo
    def select_action(self, game_board:GameBoard)->int:
        if game_board.is_first_player_turn():
            alpha = self.predictor_first
            beta = self.predictor_second
        else:
            alpha = self.predictor_second
            beta = self.predictor_first
        root_node = PolicyValueNetworkMctsNode(
                game_board=game_board,
                is_root=True,
                predictor_alpha=alpha,
                predictor_beta=beta,
                policy=0.0,
                node_selector=self.node_selector,
                child_node_selector=self.node_selector)
        selected = self.mcts(root_node=root_node)
        policies = self.mcts.get_action_rations(root_node=root_node)
        self.last_policies = policies
        self.last_action = selected
        return selected
    def get_last_policies(self):
        return self.last_policies
    def get_last_action(self):
        return self.last_action

class SelfplayNetworkBrain(SelfplayBrain):
    def __init__(self,
                predictor_first : Predictor,
                predictor_second : Predictor,
                brain_param:BrainParameter,
                exploration_param:ExplorationParameter):
        super().__init__(brain_param=brain_param)
        self.mcts = MonteCarloTreeSearcher(brain_param=brain_param)
        self.predictor_first = predictor_first
        self.predictor_second = predictor_second
        self.last_policies = None
        self.last_action = None
        self.root_node_selector = WithDirichletPolicyUCTNextNodeSelector(c_base=exploration_param.c_base, c_init=exploration_param.c_init, alpha=exploration_param.alpha, epsilon=exploration_param.epsilon)
        self.child_node_selector = PolicyUCTNextNodeSelector(c_base=exploration_param.c_base, c_init=exploration_param.c_init)
    def get_name(self):
        return 'SelfplayNetworkBrain MCTS={0} selector={1}'.format(self.network_mcts, self.policy_selector)
    def select_action(self, game_board : GameBoard)->int:
        if game_board.is_first_player_turn():
            alpha = self.predictor_first
            beta = self.predictor_second
        else:
            alpha = self.predictor_second
            beta = self.predictor_first
        root_node = PolicyValueNetworkMctsNode(
                game_board=game_board,
                is_root=True,
                predictor_alpha=alpha,
                predictor_beta=beta,
                policy=0.0,
                node_selector=self.root_node_selector,
                child_node_selector=self.child_node_selector)
        selected = self.mcts(root_node=root_node)
        policies = self.mcts.get_action_rations(root_node=root_node)
        self.last_policies = policies
        self.register_policies(game_board=game_board, policies=policies)
        self.last_action = selected
        return selected
    
    def get_last_policies(self):
        return self.last_policies
    def get_last_action(self):
        return self.last_action


class NetworkBrainFactory:
    @staticmethod
    def create_network_brain(predictor : Predictor, brain_param: BrainParameter, exploration_param:ExplorationParameter)->NetworkBrain:
        return NetworkBrain(
                predictor_first=predictor,
                predictor_second=predictor,
                brain_param=brain_param,
                exploration_param=exploration_param)
    def create_dualmodel_network_brain(predictor_first : Predictor, predictor_second : Predictor, brain_param: BrainParameter, exploration_param:ExplorationParameter)->NetworkBrain:
        return NetworkBrain(
                predictor_first=predictor_first,
                predictor_second=predictor_second,
                brain_param=brain_param,
                exploration_param=exploration_param)
    def create_selfplay_network_brain(predictor : Predictor, brain_param: BrainParameter, exploration_param:ExplorationParameter)->SelfplayNetworkBrain:
        return SelfplayNetworkBrain(
                predictor_first=predictor,
                predictor_second=predictor,
                brain_param=brain_param,
                exploration_param=exploration_param)
    def create_selfplay_dualmodel_network_brain(predictor_first : Predictor, predictor_second : Predictor, brain_param: BrainParameter, exploration_param:ExplorationParameter)->SelfplayNetworkBrain:
        return SelfplayNetworkBrain(
                predictor_first=predictor_first,
                predictor_second=predictor_second,
                brain_param=brain_param,
                exploration_param=exploration_param)

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