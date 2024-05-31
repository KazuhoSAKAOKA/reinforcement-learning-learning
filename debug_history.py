import numpy as np

from game_board import GameBoard
from stone_game_board import StoneGameBoard
from gomoku_board import GomokuBoard

from network_common import load_data_file_name


def history_view(game_board: StoneGameBoard, history_file: str):
    history = load_data_file_name(history_file)

    for (x, y_policy, y_value) in history:
        print("====== hisotry ======")
        game_board.self_cells = np.array(x[0]).reshape(game_board.board_size, game_board.board_size)
        game_board.enemy_cells = np.array(x[1]).reshape(game_board.board_size, game_board.board_size)
        game_board.turn = np.sum(x[0][:]) + np.sum(x[1][:])
        print(game_board)
        print('policy={}'.format(y_policy))
        print('value={}'.format(y_value))
        print("~~~~~~~~~~~~~~~~~~~~~")

def history_save(game_board: StoneGameBoard, history_file: str, save_file: str):
    history = load_data_file_name(history_file)
    with open(save_file, 'w') as f:
        for (x, y_policy, y_value) in history:
            f.write("====== hisotry ======\n")
            game_board.self_cells = np.array(x[0]).reshape(game_board.board_size, game_board.board_size)
            game_board.enemy_cells = np.array(x[1]).reshape(game_board.board_size, game_board.board_size)
            game_board.turn = np.sum(x[0][:]) + np.sum(x[1][:])
            f.write(str(game_board))
            f.write('\n')
            f.write('policy={}\n'.format(y_policy))
            f.write('value={}\n'.format(y_value))
            f.write("~~~~~~~~~~~~~~~~~~~~~\n")



if __name__ == '__main__':
    game_board = GomokuBoard(9)
    #history_save(GomokuBoard(11), './data/gomoku_11/first/20240511082326.history', 'first_history.txt')
    #history_save(GomokuBoard(11), './data/gomoku_11/second/20240511082326.history', 'second_history.txt')

    history_save(game_board, './data/gomoku_9/first/20240515051822.history', 'first_history.txt')
    history_save(game_board, './data/gomoku_9/second/20240515051822.history', 'second_history.txt')