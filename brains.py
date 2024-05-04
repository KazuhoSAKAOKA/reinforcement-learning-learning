from abc import abstractmethod
import numpy as np
from game_board import GameBoard

class Brain:
    @abstractmethod
    def get_name(self):
        pass
    @abstractmethod
    def select_action(self, board : GameBoard)->int:
        pass
    def __repr__(self) -> str:
        return self.get_name()

class ConsoleDebugBrain(Brain):
    def __init__(self):
        super().__init__()
    def get_name(self):
        return "ConsoleDebugBrain"
    def select_action(self, board : GameBoard)->int:
        print("First player's turn" if board.is_first_player_turn() else "Second player's turn")
        print(board)
        actions = board.get_legal_actions()
        print(actions)
        while True:
            action = int(input())
            if action in actions:
                return action

class RandomBrain(Brain):
    def __init__(self):
        super().__init__()
    def get_name(self):
        return "RandomBrain"
    def select_action(self, board : GameBoard)->int:
        actions = board.get_legal_actions()
        return np.random.choice(actions)