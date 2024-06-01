import unittest
from game_board import GameBoard, GameRelativeResult
from tictactoe_board import TicTacToeBoard
from self_play_brain import SelfplayRandomMCTSBrain
from parameter import Parameter, HistoryUpdateType
from debug_history import history_list_save

class TestStoneGameBoard(unittest.TestCase):
    def test_augmente(self):
        first = TicTacToeBoard(5)
        param = Parameter(mcts_evaluate_count=100, mcts_expand_limit=10, history_update_type=HistoryUpdateType.zero_to_one)
        brain = SelfplayRandomMCTSBrain(param)
        selected_1 = brain.select_action(first)
        step1, r = first.transit_next(selected_1)
        self.assertTrue(r)
        selected_2 = brain.select_action(step1)
        step2, r = step1.transit_next(selected_2)
        self.assertTrue(r)
        selected_3 = brain.select_action(step2)
        step3, r = step1.transit_next(selected_3)
        self.assertTrue(r)

        brain.update_history(1.0)
        history = brain.history

        new_history = first.augmente_data(history.copy())
        history_list_save(game_board=first, history=new_history, save_file='.tests/historyaugmente_history.txt')

