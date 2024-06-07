import numpy as np

from game_board import GameBoard
from stone_game_board import StoneGameBoard
from gomoku_board import GomokuBoard
from tictactoe_board import TicTacToeBoard
from self_play import HistoryData


def history_view(game_board: StoneGameBoard, history_folder: str):
    history_data = HistoryData(history_folder)
    history = history_data.deserialize()
    for (x, y_policy, y_value) in history:
        print("====== hisotry ======")
        game_board.self_cells = np.array(x[0]).reshape(game_board.board_size, game_board.board_size)
        game_board.enemy_cells = np.array(x[1]).reshape(game_board.board_size, game_board.board_size)
        game_board.turn = np.sum(x[0][:]) + np.sum(x[1][:])
        print(game_board)
        print('policy={}'.format(y_policy))
        print('value={}'.format(y_value))
        print("~~~~~~~~~~~~~~~~~~~~~")

def history_list_save(game_board: StoneGameBoard, history:list, save_file: str):
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

def history_save(game_board: StoneGameBoard, history_folder: str, save_file: str):
    history_data = HistoryData(history_folder)
    history = history_data.deserialize()
    history_list_save(game_board, history, save_file)

if __name__ == '__main__':
    game_board = TicTacToeBoard()
    #history_save(GomokuBoard(11), './data/gomoku_11/first/20240511082326.history', 'first_history.txt')
    #history_save(GomokuBoard(11), './data/gomoku_11/second/20240511082326.history', 'second_history.txt')
    history_save(game_board, '/home/kazuho/python/reinforcement-learning-learning/test_files/tictactoe_network/20240602153945.history', 'test_history.txt')
