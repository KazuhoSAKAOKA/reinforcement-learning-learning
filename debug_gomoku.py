from game import GameBoard
from gomoku_board import GomokuBoard
from agent import Agent
from network_brain import NetworkBrain
from gomoku_network import create_brain_parameter
from gui import HumanGuiBrain, run_gui
import tensorflow as tf
import os
import concurrent.futures
from threadsafe_dict import ThreadSafeDict
from self_play_brain import SelfplayBrain
from network_brain import NetworkBrain, SelfplayNetworkBrain, NetworkBrainFactory
from self_play import self_play_impl
from predictor import DualNetworkPredictor
from parameter import NetworkParameter, SelfplayParameter, BrainParameter, ExplorationParameter, InitSelfplayParameter, NetworkType, HistoryUpdateType, ActionSelectorType, judge_stats

def debug_gui(board_size : int=15):

    board = GomokuBoard(board_size=board_size)
    first_model = tf.keras.models.load_model('/home/kazuho/python/reinforcement-learning-learning/model/gomoku_11/best_first.keras')
    second_model = tf.keras.models.load_model('/home/kazuho/python/reinforcement-learning-learning/model/gomoku_11/best_second.keras')
    ts_dict = ThreadSafeDict()
    best_predictor_first = DualNetworkPredictor(model=first_model, ts_dict=ts_dict)
    best_predictor_second = DualNetworkPredictor(model=second_model, ts_dict=ts_dict)
    best_brain = NetworkBrainFactory.create_dualmodel_network_brain(
            predictor_first= best_predictor_first,
            predictor_second= best_predictor_second,
            brain_param=create_brain_parameter(11),
            exploration_param=ExplorationParameter())
    first_agent = Agent(brain=best_brain, name='best1')
    second_agent = Agent(brain=best_brain, name='best2')
    run_gui(board=board, first_agent=first_agent, second_agent=second_agent)

    
    del first_model
    del second_model
    tf.keras.backend.clear_session()
   


#first_best_model_file = './test1_model.keras'
#second_best_model_file = './test2_model.keras'
    #if os.path.exists(first_best_model_file):
    #    os.remove(first_best_model_file)
    #if os.path.exists(second_best_model_file):
    #    os.remove(second_best_model_file)
#debug_init_train(11,first_best_model_file,second_best_model_file)
#debug_selfplay(11,first_best_model_file,second_best_model_file)

#validate_network()
 
debug_gui(11)

#train_cycle_gomoku(board_size=9, brain_evaluate_count=5, selfplay_repeat=10, epoch_count=1, cycle_count=1, eval_count=1, use_cache=True)

#train_cycle_dualmodel_gomoku(board_size=15, brain_evaluate_count=500, selfplay_repeat=100, epoch_count=100, cycle_count=4, eval_count=10, use_cache=True, initial_train_count=500, initial_selfplay_repeat=1000, history_updater=ZeroToOneHistoryUpdater())
#train_cycle_dualmodel_gomoku(board_size=11, brain_evaluate_count=100, selfplay_repeat=2, epoch_count=1, cycle_count=1, eval_count=0, use_cache=True)



#train_cycle_gomoku(board_size=9, brain_evaluate_count=20, selfplay_repeat=2, epoch_count=2, cycle_count=2, eval_count=1, use_cache=True, initial_train_count=10, initial_selfplay_repeat=10, history_updater=ZeroToOneHistoryUpdater())

#train_cycle_dualmodel_gomoku(board_size=9, brain_evaluate_count=300, selfplay_repeat=100, epoch_count=50, cycle_count=10, eval_count=10, use_cache=True, initial_train_count=500, initial_selfplay_repeat=1000, history_updater=ZeroToOneHistoryUpdater())

