from game import GameBoard
from gomoku_board import GomokuBoard
from agent import Agent
from network_common import NetworkBrain,DualModelNetworkBrain, load_model
from gomoku_network import get_model_file_best, get_model_file_best_first, get_model_file_best_second
from gui import HumanGuiBrain, run_gui
from tensorflow.keras import backend as K


def debug_gui(board_size : int=15):

    board = GomokuBoard(board_size=board_size)
    first_model = load_model(get_model_file_best_first(board_size= board_size))
    second_model = load_model(get_model_file_best_second(board_size= board_size))

    #first_agent = Agent(DualModelNetworkBrain(temperature=0, evaluate_count=10, first_model=first_model, second_model=second_model))
    first_agent = Agent(HumanGuiBrain())
    second_agent = Agent(HumanGuiBrain())
    run_gui(board=board, first_agent=first_agent, second_agent=second_agent)

    K.clear_session()
    del first_model
    del second_model
    
    
debug_gui(9)

