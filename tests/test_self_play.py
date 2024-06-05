import unittest
from self_play_brain import SelfplayBrain, SelfplayRandomMCTSBrain
from self_play import self_play_impl,self_play,load_data_file_name,self_play_dualmodel,HistoryData
from tictactoe_board import TicTacToeBoard
from parameter import HistoryUpdateType, BrainParameter, SelfplayParameter
from debug_history import history_save
from datetime import datetime
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
        history_data = self_play_impl(
                first_brain=first_brain,
                second_brain=second_brain,
                game_board=TicTacToeBoard(),
                selfplay_param=test_selfplay_param)

        history=history_data.deserialize()
        #self.assertGreater(len(history), 100)
        save_file = history_folder + "/history.txt"
        #history_save(game_board=TicTacToeBoard(), history_file=history_files[0], save_file=save_file)


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

        history_data = self_play_impl(
                first_brain=first_brain,
                second_brain=second_brain,
                game_board=TicTacToeBoard(),
                selfplay_param=test_selfplay_param)

        history_first=history_data.get_primary().deserialize()
        history_second=history_data.get_secondary().deserialize()
        #self.assertGreater(len(history_first), 100)
        #history_second=load_data_file_name(history_file=history_files[1])
        #self.assertGreater(len(history_second), 100)
        #history_save(game_board=TicTacToeBoard(), history_file=history_files[0], save_file=history_first_folder + '/history.txt')
        #history_save(game_board=TicTacToeBoard(), history_file=history_files[1], save_file=history_second_folder + '/history.txt')

class TestHistoryData(unittest.TestCase):
    def test_history_data(self):
        test_folder = './test_files/history_data'
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)
        t = datetime.now()
        test_history_folder = test_folder + '/{}{}{}_{}{}{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)
        target = HistoryData(test_history_folder)
        self.assertTrue(os.path.exists(test_history_folder))
        self.assertTrue(os.path.isdir(test_history_folder))

        target.serialize([1,2,3], [4,5,6])
        self.assertEqual(target.count, 1)
        expect_file= test_history_folder + '/0000.history'
        self.assertTrue(os.path.exists(expect_file))
        self.assertTrue(os.path.isfile(expect_file))

