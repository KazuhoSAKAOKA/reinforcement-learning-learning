import unittest
from network_common import train_network
from tictactoe_board import TicTacToeBoard
from tictactoe_network import dual_network,train_cycle_tictactoe, TICTACTOE_NETWORK_PARAM, TICTACTOE_NETWORK_PARAM_DUAL,TICTACTOE_SELFPLAY_PARAM,TICTACTOE_SELFPLAY_PARAM_DUAL,TICTACTOE_BRAIN_PARAM,TICTACTOE_INIT_TRAIN_PARAM
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
    def test_model_simple(self):
        init_test_folder()
        test_model_file = TEST_FOLDER + 'model_best.keras'
        if not dual_network(test_model_file):
            self.fail('dual_network failed')
        if not os.path.exists(test_model_file):
            self.fail('model file not found')
        model = tf.keras.models.load_model(test_model_file)
        if model is None:
            self.fail('model load failed')
        game_board = TicTacToeBoard(3)
        x = game_board.reshape_to_input()
        y = model.predict(x)
        print(y)

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
                initial_selfplay_param=None,
                exploration_param=ExplorationParameter())
               
    def test_train_network_dual(self):
        init_test_folder()
        test_model_file = TEST_FOLDER + 'model_best.keras'
        train_cycle_tictactoe(
                board_size=3,
                best_model_file=test_model_file,
                history_folder=TEST_FOLDER,
                brain_evaluate_count=10,
                selfplay_repeat = 10,
                epoch_count = 5,
                cycle_count = 2,
                eval_count = 5,
                eval_judge = lambda x: True,
                use_cache = True,
                initial_selfplay_repeat = 10,
                initial_train_count = 10,
                param = Parameter(),
                is_continue = False,
                start_index = 0)
