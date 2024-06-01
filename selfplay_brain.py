from abc import abstractmethod
from enum import Enum
from game_board import GameBoard
from parameter import Parameter, HistoryUpdaterType
import numpy as np
from brains import Brain
from mcts_node import MonteCarloTreeSearcher,MaxActionSelector,RandomMctsNode,UCB1NodeSelector


class HistoryUpdater:
    def __call__(self, history:list, value:float)->None:
        i = len(history) - 1
        while i >= 0:
            history[i][2] = value
            i -= 1

class DiscountHistoryUpdater(HistoryUpdater):
    def __init__(self, gamma:float):
        self.gamma = gamma
    def __call__(self, history:list, value:float)->None:
        i = len(history) - 1
        while i >= 0:
            history[i][2] = value
            value = self.gamma * value
            i -= 1


# ゲーム開始時の価値が0で、勝利時に+-1になるようにする
class ZeroToOneHistoryUpdater(HistoryUpdater):
    def __call__(self, history:list, value:float)->None:
        slope = value * (1.0 / (len(history) - 1))
        for i in range(0, len(history)):
            history[i][2] = slope * i



class HistoryUpdaterFactory:
    @abstractmethod
    def create_history_updater(param: Parameter)->HistoryUpdater:
        if param.history_update_type == HistoryUpdaterType.constants:
            return HistoryUpdater()
        elif param.history_update_type == HistoryUpdaterType.pram_gamma:
            return DiscountHistoryUpdater(param.gamma)
        elif param.history_update_type == HistoryUpdaterType.zero_to_one:
            return ZeroToOneHistoryUpdater()
        Exception('Unknown type')
class SelfplayBrain(Brain):
    def __init__(self, param:Parameter):
        super().__init__()
        self.history = []
        self.history_updater = HistoryUpdaterFactory.create_history_updater(param)
    def get_name(self):
        return "SelfplayBrain"
    def register_policies(self, game_board : GameBoard, policies: list):
        if not game_board.is_ignore_state():
            self.history.append([game_board.to_hisotry_record(), policies, None])  
    def update_history(self, value: float):
        self.history_updater(history=self.history, value=value)
    def reset(self):
        self.history = []


class SelfplayRandomBrain(SelfplayBrain):
    def __init__(self, param:Parameter):
        super().__init__(param=param)
    def get_name(self):
        return "SelfplayRandomBrain"
    def select_action(self, board : GameBoard)->int:
        legal_actions = board.get_legal_actions()
        selected = np.random.choice(legal_actions)
        ratios = np.zeros(board.get_output_size(), dtype=np.float32)
        base_ratio = 1.0 / len(legal_actions + 1)
        for action in legal_actions:
            if selected == action:
                ratios[action] = base_ratio * 2
            else:
                ratios[action] = base_ratio

        self.register_policies(board, ratios)
        return selected


class SelfplayRandomMCTSBrain(SelfplayBrain):
    def __init__(self, param:Parameter):
        super().__init__(param=param)
        self.mcts = MonteCarloTreeSearcher(
            evaluate_count=param.mcts_evaluate_count,
            expand_limit=param.mcts_expand_limit,
            action_selector=MaxActionSelector())
    def get_name(self):
        return "SelfplayRandomMCTSBrain"
    def select_action(self, game_board : GameBoard)->int:
        root_node = RandomMctsNode(
            game_board=game_board,
            is_root=True,
            expand_limit=self.expand_limit,
            node_selector=UCB1NodeSelector())
        selected = self.mcts(root_node)
        ratios = self.mcts.get_action_rations(root_node)
        self.register_policies(game_board=game_board, policies=ratios)
        return selected