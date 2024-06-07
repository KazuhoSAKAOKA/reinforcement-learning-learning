import unittest
from network_common import train_network
from tictactoe_board import TicTacToeBoard
from tictactoe_network import create_dual_network_model,train_cycle_tictactoe, TICTACTOE_NETWORK_PARAM, TICTACTOE_NETWORK_PARAM_DUAL,TICTACTOE_SELFPLAY_PARAM,TICTACTOE_SELFPLAY_PARAM_DUAL,TICTACTOE_BRAIN_PARAM,TICTACTOE_INIT_TRAIN_PARAM
from parameter import BrainParameter, NetworkParameter, SelfplayParameter, InitSelfplayParameter,ExplorationParameter
import os
import shutil
import copy
import tensorflow as tf

TEST_FOLDER = './test_files/tictactoe_network/'
def init_test_folder():
    if os.path.exists(TEST_FOLDER):
        shutil.rmtree(TEST_FOLDER)
    os.makedirs(TEST_FOLDER)

class TestTicTacToeNetwork(unittest.TestCase):
#    def test_model_1(self):
#        test_model_file = TEST_FOLDER + 'model_best.keras'
#        model = tf.keras.models.load_model(test_model_file)
#        game_board = TicTacToeBoard(3)
#        x = game_board.reshape_to_input()
#        y = model.predict(x)
#        print(y)

    def test_model_simple(self):
        model = create_dual_network_model()
        game_board = TicTacToeBoard(3)
        x = game_board.reshape_to_input()
        y = model.predict(x)
        print(y)
        del model
        tf.keras.backend.clear_session()
        
    def test_train_network(self):
        init_test_folder()
        test_model_file = TEST_FOLDER + 'model_best.keras'
        test_network_param = copy.copy(TICTACTOE_NETWORK_PARAM)
        test_network_param.best_model_file = test_model_file
        test_brain_param = copy.copy(TICTACTOE_BRAIN_PARAM)
        test_brain_param.mcts_evaluate_count = 5
        test_brain_param.mcts_expand_limit = 2
        test_selfplay_param = copy.copy(TICTACTOE_SELFPLAY_PARAM)
        test_selfplay_param.history_folder = TEST_FOLDER
        test_selfplay_param.selfplay_repeat = 10
        test_selfplay_param.cycle_count = 2
        test_selfplay_param.evaluate_count = 5
        test_selfplay_param.train_epoch = 5
        test_init_selfplay_param = copy.copy(TICTACTOE_INIT_TRAIN_PARAM)
        test_init_selfplay_param.selfplay_repeat = 10
        test_init_selfplay_param.train_epoch = 5
        train_cycle_tictactoe(
                network_param=test_network_param,
                brain_param=test_brain_param,
                selfplay_param=test_selfplay_param,
                initial_selfplay_param=test_init_selfplay_param,
                exploration_param=ExplorationParameter())
               
    def test_train_network_dual(self):
        init_test_folder()
        test_model_first_file = TEST_FOLDER + '/first/model_best_first.keras'
        test_model_second_file = TEST_FOLDER + '/second/model_best_second.keras'
        test_network_param = copy.copy(TICTACTOE_NETWORK_PARAM_DUAL)
        test_network_param.best_model_file = test_model_first_file
        test_network_param.best_model_file_second = test_model_second_file
        test_brain_param = copy.copy(TICTACTOE_BRAIN_PARAM)
        test_brain_param.mcts_evaluate_count = 5
        test_brain_param.mcts_expand_limit = 2
        test_selfplay_param = copy.copy(TICTACTOE_SELFPLAY_PARAM_DUAL)
        test_selfplay_param.history_folder = TEST_FOLDER + '/first'
        test_selfplay_param.is_dual_model = True
        test_selfplay_param.selfplay_repeat = 10
        test_selfplay_param.cycle_count = 2
        test_selfplay_param.evaluate_count = 5
        test_selfplay_param.train_epoch = 5
        test_init_selfplay_param = copy.copy(TICTACTOE_INIT_TRAIN_PARAM)
        test_init_selfplay_param.selfplay_repeat = 10
        test_init_selfplay_param.train_epoch = 5
        train_cycle_tictactoe(
                network_param=test_network_param,
                brain_param=test_brain_param,
                selfplay_param=test_selfplay_param,
                initial_selfplay_param=test_init_selfplay_param,
                exploration_param=ExplorationParameter())
