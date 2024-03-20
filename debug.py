from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from network_common import load_model,predict,NetworkBrain
from tictactoe_network import MODEL_FILE_BEST
from tictactoe_board import TicTacToeBoard
from mini_max import AlphaBetaBrain
from agent import Agent
from game import GameEnv
import numpy as np
import pv_mcts



def output_step(board :TicTacToeBoard, model :Model):
    p, v = predict(model, board)
    print(board)
    print(p)
    print(v)

def degut_predict_only(model):
    board = TicTacToeBoard()
    output_step(board, model)

    _, board = board.transit_next(4)
    output_step(board, model)

    _, board = board.transit_next(1)
    output_step(board, model)

    _, board = board.transit_next(2)
    output_step(board, model)

    _, board = board.transit_next(6)
    output_step(board, model)

    _, board = board.transit_next(5)
    output_step(board, model)


def output_mcts_step(board :TicTacToeBoard, brain: NetworkBrain):
    board = step(board, [4,1,2,6])
    v = brain.select_action(board)
    print(board)
    print(brain.get_last_policies())

def debug_predict(_1, board : TicTacToeBoard):
    legal_actions = board.get_legal_actions()
    p = 1.0 / len(legal_actions)
    return np.full(shape=(len(legal_actions),), fill_value=p), 0.0
def step(board :TicTacToeBoard, steps : list) -> TicTacToeBoard:
    for action in steps:
        _, board = board.transit_next(action)
    return board

def debug_mcts(model):
    board = TicTacToeBoard()
    brain = NetworkBrain(0.1, 5000, lambda x: debug_predict(model, x), lambda x: debug_predict(model, x))
    #output_mcts_step(board, brain)

    board = step(board, [4,1,2,6])
    p = predict(model, board)
    print(p)
    output_mcts_step(board, brain)

#    _, board = board.transit_next(4)
#    output_mcts_step(board, brain)
#    _, board = board.transit_next(1)
#    output_mcts_step(board, brain)
#    _, board = board.transit_next(2)
#    output_mcts_step(board, brain)
#    _, board = board.transit_next(6)
#    output_mcts_step(board, brain)
#    _, board = board.transit_next(5)
#    output_mcts_step(board, brain)

def debug_play(model):
    board = TicTacToeBoard()
    brain_agent = Agent(NetworkBrain(0, 50, lambda x: predict(model, x), lambda x: predict(model, x)))
    alpha_beta_agent = Agent(AlphaBetaBrain())
    #env = GameEnv(board, brain_agent, alpha_beta_agent)
    #env.play_n(5)
    env = GameEnv(board, alpha_beta_agent, brain_agent)
    env.play_n(5)



file = MODEL_FILE_BEST
#file = './model/tictactoe/20240314225839.keras'
model = load_model(MODEL_FILE_BEST)
debug_play(model)

#degut_predict_only(model)
#debug_mcts(model)


K.clear_session()
del model