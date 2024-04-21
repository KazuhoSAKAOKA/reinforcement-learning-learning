from game_board import GameBoard
from parameter import PARAM
import numpy as np

class SelfplayBrain:
    def __init__(self):
        self.history = []
    def get_name(self):
        return "SelfplayBrain"
    def register_action(self, board : GameBoard, selected: int):
        policies = [0.0] * board.get_output_size()
        policies[selected] = 1.0
        self.history.append([board.get_model_state(), policies, None])  
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
        action = np.random.choice(board.get_legal_actions())
        self.register_action(board, action)
        return action
