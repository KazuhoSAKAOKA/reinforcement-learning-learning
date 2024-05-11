from abc import abstractmethod
from enum import Enum
from game_board import GameBoard
from parameter import PARAM
import numpy as np
from brains import Brain


class HistoryUpdater:
    def __call__(self, history:list, value:float)->None:
        i = len(history) - 1
        while i >= 0:
            history[i][2] = value
            value = PARAM.gamma * value
            i -= 1

# ゲーム開始時の価値が0で、勝利時に+-1になるようにする
class ZeroToOneHistoryUpdater(HistoryUpdater):
    def __call__(self, history:list, value:float)->None:
        slope = value * (1.0 / (len(history) - 1))
        for i in range(0, len(history)):
            history[i][2] = slope * i

class HistoryUpdaterType(Enum):
    pram_gamma = 1
    zero_to_one = 2

class HistoryUpdaterFactory:
    @abstractmethod
    def create_history_updater(t:HistoryUpdaterType):
        if t == HistoryUpdaterType.pram_gamma:
            return HistoryUpdater()
        elif t == HistoryUpdaterType.zero_to_one:
            return ZeroToOneHistoryUpdater()
        Exception('Unknown type')
class SelfplayBrain(Brain):
    def __init__(self, history_updater:HistoryUpdater):
        super().__init__()
        self.history = []
        self.history_updater = history_updater
    def get_name(self):
        return "SelfplayBrain"
    def register_policies(self, game_board : GameBoard, policies: list):
        if not game_board.is_ignore_state():
            self.history.append([game_board.to_hisotry_record(), policies, None])  
    def update_history(self, value):
        self.history_updater(history=self.history, value=value)
    def reset(self):
        self.history = []


class SelfplayRandomBrain(SelfplayBrain):
    def __init__(self, history_updater:HistoryUpdater):
        super().__init__(history_updater=history_updater)
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
