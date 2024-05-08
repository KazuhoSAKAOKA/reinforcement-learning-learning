from game_board import GameBoard
from parameter import PARAM
import numpy as np
from brains import Brain

class SelfplayBrain(Brain):
    def __init__(self):
        super().__init__()
        self.history = []
    def get_name(self):
        return "SelfplayBrain"
    def register_policies(self, board : GameBoard, policies: list):
        self.history.append([board.to_hisotry_record(), policies, None])  
    def update_history(self, value):
        i = len(self.history) - 1
        while i >= 0:
            self.history[i][2] = value
            value = PARAM.gamma * value
            i -= 1
    def reset(self):
        self.history = []


class SelfplayRandomBrain(SelfplayBrain):
    def __init__(self):
        super().__init__()
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
