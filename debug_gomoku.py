from game import GameBoard
from gomoku_board import GomokuBoard
from agent import Agent
from network_brain import NetworkBrain, predict, predict_core
from network_common import initial_train_model
from gomoku_network import dual_network, get_model_file_best, get_model_file_best_first, get_model_file_best_second, train_cycle_gomoku,train_cycle_dualmodel_gomoku,get_history_folder_first,get_history_folder_second
from gui import HumanGuiBrain, run_gui
import tensorflow as tf
import os
import concurrent.futures
from threadsafe_dict import ThreadSafeDict
from network_brain import SelfplayDualModelNetworkBrain
from self_play import self_play_dualmodel
from parameter import PARAM

def debug_gui(board_size : int=15):

    board = GomokuBoard(board_size=board_size)
    first_model = tf.keras.models.load_model(get_model_file_best_first(board_size= board_size))
    second_model = tf.keras.models.load_model(get_model_file_best_second(board_size= board_size))

    #first_agent = Agent(DualModelNetworkBrain(temperature=0, evaluate_count=10, first_model=first_model, second_model=second_model))
    first_agent = Agent(HumanGuiBrain())
    second_agent = Agent(HumanGuiBrain())
    run_gui(board=board, first_agent=first_agent, second_agent=second_agent)

    
    del first_model
    del second_model
    tf.keras.backend.clear_session()
   

def validate_network():
    test_file = './model/gomoku_11/test_model.keras'
    os.remove(test_file)
    dual_network(test_file, 11)
    model = tf.keras.models.load_model(test_file)

    def output(board: GameBoard):
        print(board)
        policies, v = predict_core(model, board)
        print('policy={}'.format(policies))
        print('value={}'.format(v))

    board = GomokuBoard(11)
    output(board)

    board, _ = board.transit_next(60)
    output(board)

    board, _ = board.transit_next(0)
    output(board)

    board, _ = board.transit_next(1)
    output(board)

    del model
    tf.keras.backend.clear_session()

def debug_selfplay(board_size:int = 11):
    first_best_model_file = './test1_model.keras'
    second_best_model_file = './test2_model.keras'
    #if os.path.exists(first_best_model_file):
    #    os.remove(first_best_model_file)
    #if os.path.exists(second_best_model_file):
    #    os.remove(second_best_model_file)
    debug_file = '.DEBUG_OUT.txt'
    if os.path.exists(debug_file):
        os.remove(debug_file)

    PARAM.alpha = 0.0
    ts_dict = ThreadSafeDict()
    brain_evaluate_count = 20
    #dual_network(first_best_model_file, board_size, True)
    #dual_network(second_best_model_file, board_size, True)

#    initial_train_model(game_board=GomokuBoard(board_size=board_size), 
#                        first_best_model_file=first_best_model_file, 
#                        second_best_model_file=second_best_model_file,
#                        history_first_folder=get_history_folder_first(board_size),
#                        history_second_folder=get_history_folder_second(board_size),
#                        initial_selfplay_repeat=100,
#                        initial_train_count=100,
#                        executor=concurrent.futures.ThreadPoolExecutor(2))

    first_model = tf.keras.models.load_model(first_best_model_file)
    second_model = tf.keras.models.load_model(second_best_model_file)
    first_brain = SelfplayDualModelNetworkBrain(brain_evaluate_count, first_model, second_model, ts_dict)
    second_brain = SelfplayDualModelNetworkBrain(brain_evaluate_count, first_model, second_model, ts_dict)
    first_history_file, second_history_fine = self_play_dualmodel(first_brain, second_brain,GomokuBoard(board_size=board_size), 2, get_history_folder_first(board_size), get_history_folder_second(board_size))   

debug_selfplay()

#validate_network()
 
#debug_gui(9)

#train_cycle_gomoku(board_size=9, brain_evaluate_count=5, selfplay_repeat=10, epoch_count=1, cycle_count=1, eval_count=1, use_cache=True)

#train_cycle_dualmodel_gomoku(board_size=11, brain_evaluate_count=100, selfplay_repeat=100, epoch_count=50, cycle_count=20, eval_count=10, use_cache=True)
#train_cycle_dualmodel_gomoku(board_size=11, brain_evaluate_count=100, selfplay_repeat=2, epoch_count=1, cycle_count=1, eval_count=0, use_cache=True)
