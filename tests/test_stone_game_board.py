import unittest
from game_board import GameBoard, GameRelativeResult
from tictactoe_board import TicTacToeBoard
from self_play_brain import SelfplayRandomMCTSBrain
from parameter import Parameter, HistoryUpdateType
from debug_history import history_list_save

class TestStoneGameBoard(unittest.TestCase):
    def test_augmente(self):
        state = TicTacToeBoard(5)
        history = []
        state.self_cells[0][0] = 1
        state.enemy_cells[0][1] = 1
        state.self_cells[0][2] = 1
        state.enemy_cells[0][3] = 1
        state.self_cells[0][4] = 1
        state.enemy_cells[1][0] = 1
        history.append([state.to_hisotry_record(), 
                        [0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.01, 0.02, 0.03, 0.04,
                         0.05, 0.06, 0.07, 0.08, 0.09,
                         0.10, 0.11, 0.12, 0.13, 0.14,
                         0.15, 0.16, 0.17, 0.18, 0.19,
                         ], 0.5])

        new_history = state.augmente_data(history.copy())
        self.assertEqual(len(new_history), 1 + 7)
        history_list_save(game_board=state, history=new_history, save_file='./tests/history/augmente_history.txt')

