import unittest
from self_play_brain import SelfplayBrain, SelfplayRandomMCTSBrain
from self_play import self_play_impl,self_play,load_data_file_name,self_play_dualmodel
from tictactoe_board import TicTacToeBoard
from parameter import HistoryUpdateType, BrainParameter, SelfplayParameter
from debug_history import history_save
import os
import shutil

class TestSelfPlay(unittest.TestCase):
    def test_self_play(self):
        history_folder = './test_files/history'
        if os.path.exists(history_folder):
            shutil.rmtree(history_folder)
        os.makedirs(history_folder)
        brain_param = BrainParameter(mcts_evaluate_count=10, mcts_expand_limit=10, history_update_type=HistoryUpdateType.zero_to_one)
        first_brain=SelfplayRandomMCTSBrain(brain_param=brain_param)
        second_brain= SelfplayRandomMCTSBrain(brain_param=brain_param)
        test_selfplay_param = SelfplayParameter(history_folder=history_folder, selfplay_repeat=100)
        history_files = self_play_impl(
                first_brain=first_brain,
                second_brain=second_brain,
                game_board=TicTacToeBoard(),
                selfplay_param=test_selfplay_param)

        history=load_data_file_name(history_file=history_files[0])
        self.assertGreater(len(history), 100)
        save_file = history_folder + "/history.txt"
        history_save(game_board=TicTacToeBoard(), history_file=history_files[0], save_file=save_file)


    def test_self_play_dualmodel(self):
        history_parent = './test_files/'
        if os.path.exists(history_parent):
            shutil.rmtree(history_parent)

        history_first_folder = history_parent + '/history_first'
        history_second_folder = history_parent + '/history_second'
        os.makedirs(history_first_folder)
        os.makedirs(history_second_folder)

        brain_param = BrainParameter(mcts_evaluate_count=10, mcts_expand_limit=10, history_update_type=HistoryUpdateType.zero_to_one)
        first_brain=SelfplayRandomMCTSBrain(brain_param=brain_param)
        second_brain= SelfplayRandomMCTSBrain(brain_param=brain_param)
        test_selfplay_param = SelfplayParameter(history_folder=history_first_folder, history_folder_second=history_second_folder, selfplay_repeat=100)
        brain_param = BrainParameter(mcts_evaluate_count=10, mcts_expand_limit=10, history_update_type=HistoryUpdateType.zero_to_one)
        first_brain=SelfplayRandomMCTSBrain(brain_param=brain_param)
        second_brain= SelfplayRandomMCTSBrain(brain_param=brain_param)

        history_files = self_play_impl(
                first_brain=first_brain,
                second_brain=second_brain,
                game_board=TicTacToeBoard(),
                selfplay_param=test_selfplay_param)

        history_first=load_data_file_name(history_file=history_files[0])
        self.assertGreater(len(history_first), 100)
        history_second=load_data_file_name(history_file=history_files[1])
        self.assertGreater(len(history_second), 100)

        history_save(game_board=TicTacToeBoard(), history_file=history_files[0], save_file=history_first_folder + '/history.txt')
        history_save(game_board=TicTacToeBoard(), history_file=history_files[1], save_file=history_second_folder + '/history.txt')
