import unittest
from gomoku_network import train_cycle_gomoku,create_network_parameter_dual,create_selfplay_parameter_dual
from parameter import NetworkParameter,SelfplayParameter,BrainParameter,ExplorationParameter,NetworkType,HistoryUpdateType,ActionSelectorType, InitSelfplayParameter, judge_stats
import os
import shutil

TEST_FOLDER = './test_files/gomoku_network/'
def init_test_folder():
    if os.path.exists(TEST_FOLDER):
        shutil.rmtree(TEST_FOLDER)
    os.makedirs(TEST_FOLDER)

class TestGomokuNetwork(unittest.TestCase):
    def test_train_cycle_gomoku_dual(self):
        init_test_folder()
        test_model_first_file = TEST_FOLDER + '/model/first/model_best.keras'
        test_model_second_file = TEST_FOLDER + 'model//second/model_best.keras'
        test_history_first_folder = TEST_FOLDER + '/history/first/model_best.keras'
        test_history_second_folder = TEST_FOLDER + '/history/second/model_best.keras'

        test_board_size = 7
        network_parameter= create_network_parameter_dual(test_board_size)
        network_parameter.best_model_file = test_model_first_file
        network_parameter.best_model_file_second = test_model_second_file
        
        selfplay_parameter = create_selfplay_parameter_dual(test_board_size)
        selfplay_parameter.cycle_count = 2
        selfplay_parameter.train_epoch = 5
        selfplay_parameter.selfplay_repeat = 10
        selfplay_parameter.evaluate_count = 5
        brain_parameter = BrainParameter(test_board_size)
        brain_parameter.mcts_evaluate_count = 5
        brain_parameter.mcts_expand_limit = 2

        exploration_parameter = ExplorationParameter(test_board_size)
        init_selfplay_parameter = InitSelfplayParameter(test_board_size)
        init_selfplay_parameter.selfplay_repeat = 10
        init_selfplay_parameter.train_epoch = 5

        train_cycle_gomoku(
                board_size=test_board_size,
                network_param=network_parameter,
                selfplay_param=selfplay_parameter,
                brain_param=brain_parameter,
                exploration_param=exploration_parameter,
                initial_selfplay_param=init_selfplay_parameter)
        