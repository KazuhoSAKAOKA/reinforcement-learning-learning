import numpy as np


class ConsoleDebugBrain:
    def __init__(self):
        pass
    def get_name(self):
        return "ConsoleDebugBrain"
    def select_action(self, board):
        print("First player's turn" if board.is_first_player_turn() else "Second player's turn")
        print(board)
        actions = board.get_legal_actions()
        print(actions)
        while True:
            action = int(input())
            if action in actions:
                return action

class RandomBrain:
    def __init__(self):
        pass
    def get_name(self):
        return "RandomBrain"
    def select_action(self, board):
        actions = board.get_legal_actions()
        return np.random.choice(actions)