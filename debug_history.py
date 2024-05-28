import numpy as np

from game_board import GameBoard
from stone_game_board import StoneGameBoard
from tictactoe_board import TicTacToeBoard
from gomoku_board import GomokuBoard

from network_common import load_data_file_name


def history_view(board: StoneGameBoard, history_file: str):
    history = load_data_file_name(history_file)

    for (x, y_policy, y_value) in history:
        print("====== hisotry ======")
        board.self_cells = np.array(x[0]).reshape(board.board_size, board.board_size)
        board.enemy_cells = np.array(x[1]).reshape(board.board_size, board.board_size)
        board.turn = np.sum(x[0][:]) + np.sum(x[1][:])
        print(board)
        print('policy={}'.format(y_policy))
        print('value={}'.format(y_value))
        print("~~~~~~~~~~~~~~~~~~~~~")

def history_save(board: StoneGameBoard, history_file: str, save_file: str):
    history = load_data_file_name(history_file)
    with open(save_file, 'w') as f:
        for (x, y_policy, y_value) in history:
            f.write("====== hisotry ======\n")
            board.self_cells = np.array(x[0]).reshape(board.board_size, board.board_size)
            board.enemy_cells = np.array(x[1]).reshape(board.board_size, board.board_size)
            board.turn = np.sum(x[0][:]) + np.sum(x[1][:])
            f.write(str(board))
            f.write('\n')
            f.write('policy={}\n'.format(y_policy))
            f.write('value={}\n'.format(y_value))
            f.write("~~~~~~~~~~~~~~~~~~~~~\n")



if __name__ == '__main__':
    board = GomokuBoard(9)
    #history_save(GomokuBoard(11), './data/gomoku_11/first/20240511082326.history', 'first_history.txt')
    #history_save(GomokuBoard(11), './data/gomoku_11/second/20240511082326.history', 'second_history.txt')

    history_save(board, './data/gomoku_9/first/20240515051822.history', 'first_history.txt')
    history_save(board, './data/gomoku_9/second/20240515051822.history', 'second_history.txt')