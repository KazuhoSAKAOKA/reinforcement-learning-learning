from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from network_common import load_model,predict,NetworkBrain
from tictactoe_network import MODEL_FILE_BEST
from tictactoe_board import TicTacToeBoard
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
    v = brain.select_action(board)
    print(board)
    print(brain.get_last_policies())


def debug_mcts(model):
    board = TicTacToeBoard()
    brain = NetworkBrain(0.1, 50, lambda x: predict(model, x), lambda x: predict(model, x))
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


#file = MODEL_FILE_BEST
file = './model/tictactoe/20240314225839.keras'
model = load_model(MODEL_FILE_BEST)
degut_predict_only(model)
debug_mcts(model)


K.clear_session()
del model