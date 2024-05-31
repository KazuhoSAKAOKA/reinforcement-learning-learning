import unittest
from self_play_brain import SelfplayBrain, SelfplayRandomMCTSBrain
from self_play import self_play,load_data_file_name,self_play_dualmodel
from tictactoe_board import TicTacToeBoard
from parameter import Parameter,HistoryUpdateType
from debug_history import history_save
import os
import shutil

class TestSelfPlay(unittest.TestCase):
    def test_self_play(self):
        history_folder = './tests/history'
        if os.path.exists(history_folder):
            shutil.rmtree(history_folder)
        os.makedirs(history_folder)
        param = Parameter(mcts_evaluate_count=10, mcts_expand_limit=10, history_update_type=HistoryUpdateType.zero_to_one)
        first_brain=SelfplayRandomMCTSBrain(param=param)
        second_brain= SelfplayRandomMCTSBrain(param=param)
        repeat_count = 100
        history_file = self_play(
                first_brain=first_brain,
                second_brain=second_brain,
                game_board=TicTacToeBoard(),
                repeat_count=repeat_count,
                history_folder=history_folder)

        history=load_data_file_name(history_file=history_file)
        self.assertGreater(len(history), 100)
        save_file = history_folder + "/history.txt"
        history_save(game_board=TicTacToeBoard(), history_file=history_file, save_file=save_file)


    def test_self_play_dualmodel(self):
        history_parent = './tests/history'
        if os.path.exists(history_parent):
            shutil.rmtree(history_parent)

        history_first_folder = history_parent + '/history_first'
        history_second_folder = history_parent + '/history_second'
        os.makedirs(history_first_folder)
        os.makedirs(history_second_folder)
        param = Parameter(mcts_evaluate_count=10, mcts_expand_limit=10, history_update_type=HistoryUpdateType.zero_to_one)
        first_brain=SelfplayRandomMCTSBrain(param=param)
        second_brain= SelfplayRandomMCTSBrain(param=param)
        repeat_count = 100
        history_first_file, history_second_file = self_play_dualmodel(
                first_brain=first_brain,
                second_brain=second_brain,
                game_board=TicTacToeBoard(),
                repeat_count=repeat_count,
                history_folder_first=history_first_folder,
                history_folder_second=history_second_folder)

        history_first=load_data_file_name(history_file=history_first_file)
        self.assertGreater(len(history_first), 100)
        history_second=load_data_file_name(history_file=history_second_file)
        self.assertGreater(len(history_second), 100)

        history_save(game_board=TicTacToeBoard(), history_file=history_first_file, save_file=history_first_folder + '/history.txt')
        history_save(game_board=TicTacToeBoard(), history_file=history_second_file, save_file=history_second_folder + '/history.txt')
