import numpy as np

from game_board import GameBoard
from tictactoe_board import TicTacToeBoard
from network_common import load_data_file_name


def history_view(history_file: str):
    history = load_data_file_name(history_file)

    for (x, y_policy, y_value) in history:
        board = TicTacToeBoard()
        print("====== hisotry ======")
        board.self_cells = np.array(x[0]).reshape(3, 3)
        board.enemy_cells = np.array(x[1]).reshape(3, 3)
        board.turn = sum(x[0]) + sum(x[1])
        print(board)
        print(y_policy)
        print(y_value)
        print("~~~~~~~~~~~~~~~~~~~~~")




if __name__ == '__main__':
    history_view('./data/tictactoe/second/random.history')