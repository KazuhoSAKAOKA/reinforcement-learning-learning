from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import os
from game_board import GameBoard, get_first_player_value
from tictactoe_board import TicTacToeBoard
from pv_mcts import pv_mcts_scores
from game import GameEnv
from agent import Agent
from brains import RandomBrain, ConsoleDebugBrain
from mini_max import AlphaBetaBrain
from montecarlo import MonteCarloBrain
from network_common import train_network, self_play, NetworkBrain, predict, write_data, load_data, train_cycle, train_2_cycle
DN_FILTERS = 128
DN_RESIDUAL_NUM = 16
DN_INPUT_SHAPE = (3,3,2)
DN_OUTPUT_SIZE = 9

MODEL_FILE_BEST = './model/tictactoe/best.keras'
MODEL_FILE_LATEST = './model/tictactoe/latest.keras'
HISTORY_FOLDER = './data/tictactoe'

MODEL_FILE_BEST_FIRST = './model/tictactoe/first/best.keras'
MODEL_FILE_BEST_SECOND = './model/tictactoe/second/best.keras'
HISTORY_FOLDER_FIRST = './data/tictactoe/first'
HISTORY_FOLDER_SECOND = './data/tictactoe/second'
def conv(filters):
    return Conv2D(filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

def residual_block():
    def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation('relu')(x)
        return x
    return f

def dual_network(file_best=MODEL_FILE_BEST):
    if os.path.exists(file_best):
        return
    parent = os.path.dirname(file_best)
    os.makedirs(parent, exist_ok=True)
    
    input = Input(shape=DN_INPUT_SHAPE)

    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    x = GlobalAveragePooling2D()(x)

    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005), activation='softmax', name='pi')(x)

    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh', name='v')(v)

    model = Model(inputs=input, outputs=[p,v])
    #print(model.summary())

    model.save(file_best)

    K.clear_session()
    del model

RN_EPOCHS=100
def load_data():
    history_path = sorted(Path('./data/tictactoe').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)


def convert(xs):
    a,b,c = (3,3,2)
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)

    return xs

def train_network_tictactoe(epochs=RN_EPOCHS):
    train_network(MODEL_FILE_BEST, HISTORY_FOLDER, convert, epochs)


def train():
    dual_network()
    train_cycle(MODEL_FILE_BEST, HISTORY_FOLDER, TicTacToeBoard(), cycle_count=4, eval_count=10)

    board = TicTacToeBoard()
    player = Agent(ConsoleDebugBrain())
    model = load_model(MODEL_FILE_BEST)
    network_agent = Agent(NetworkBrain(0.0, 200, lambda x: predict(model, x), lambda x: predict(model, x)))
    env = GameEnv(board, player, network_agent)
    r = env.play()
    print(r)
    # モデルの破棄
    K.clear_session()
    del model

def train2():
    dual_network(MODEL_FILE_BEST_FIRST)
    dual_network(MODEL_FILE_BEST_SECOND)
    train_2_cycle(MODEL_FILE_BEST_FIRST, MODEL_FILE_BEST_SECOND, HISTORY_FOLDER_FIRST, HISTORY_FOLDER_SECOND, TicTacToeBoard(), self_play_repeat=500, epoch_count=200, cycle_count=10, eval_count=20)

    #board = TicTacToeBoard()
    #player = Agent(ConsoleDebugBrain())
    #model = load_model(MODEL_FILE_BEST)
    #network_agent = Agent(NetworkBrain(0.0, 200, lambda x: predict(model, x), lambda x: predict(model, x)))
    #env = GameEnv(board, player, network_agent)
    #r = env.play()
    #print(r)
    # モデルの破棄
    #K.clear_session()
    #del model

def play_fair(n: int, agent_1 : Agent, agent_2 : Agent):
    env = GameEnv(TicTacToeBoard(), agent_1, agent_2)
    env.play_n(n)
    env = GameEnv(TicTacToeBoard(), agent_2, agent_1)
    env.play_n(n)


def test_play():
    model = load_model(MODEL_FILE_BEST)
    network_agent = Agent(NetworkBrain(0.0, 200, lambda x: predict(model, x), lambda x: predict(model, x)))

    random_agent = Agent(RandomBrain())
    alpha_beta_agent = Agent(AlphaBetaBrain())
    montecarlo_agent = Agent(MonteCarloBrain(50))

    play_fair(20, network_agent, random_agent)
    play_fair(20, network_agent, alpha_beta_agent)
    play_fair(20, network_agent, montecarlo_agent)

    # モデルの破棄
    K.clear_session()
    del model


if __name__ == '__main__':
    train()
