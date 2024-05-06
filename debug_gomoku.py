from game import GameBoard
from gomoku_board import GomokuBoard
from agent import Agent
from network_common import NetworkBrain,DualModelNetworkBrain
from gomoku_network import get_model_file_best, get_model_file_best_first, get_model_file_best_second, train_cycle_gomoku,train_cycle_dualmodel_gomoku
from gui import HumanGuiBrain, run_gui
import tensorflow as tf

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
    
#debug_gui(9)

train_cycle_gomoku(board_size=9, brain_evaluate_count=5, selfplay_repeat=10, epoch_count=1, cycle_count=1, eval_count=1, use_cache=True)

train_cycle_dualmodel_gomoku(board_size=9, brain_evaluate_count=5, selfplay_repeat=10, epoch_count=1, cycle_count=1, eval_count=1, use_cache=True)
