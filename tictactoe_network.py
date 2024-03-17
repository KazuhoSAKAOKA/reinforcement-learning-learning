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
from montecarlo import MonteCarloBrain
from network_common import train_network, self_play, NetworkBrain, predict, write_data, load_data, train_cycle
DN_FILTERS = 128
DN_RESIDUAL_NUM = 16
DN_INPUT_SHAPE = (3,3,2)
DN_OUTPUT_SIZE = 9

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

def dual_network():
    if os.path.exists('./model/tictactoe/best.keras'):
        return
    
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

    os.makedirs('./model/tictactoe', exist_ok=True)
    model.save('./model/tictactoe/best.keras')

    K.clear_session()
    del model

RN_EPOCHS=100
def load_data():
    history_path = sorted(Path('./data/tictactoe').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

MODEL_FILE_BEST = './model/tictactoe/best.keras'
MODEL_FILE_LATEST = './model/tictactoe/latest.keras'
HISTORY_FOLDER = './data/tictactoe'

def convert(xs):
    a,b,c = (3,3,2)
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)

    return xs

def train_network_tictactoe(epochs=RN_EPOCHS):



    train_network(MODEL_FILE_BEST, HISTORY_FOLDER, convert, epochs)


if __name__ == '__main__':
    dual_network()
    train_cycle(MODEL_FILE_BEST, './model/tictactoe', HISTORY_FOLDER, TicTacToeBoard(), 500, 200, 2)

    board = TicTacToeBoard()
    player = Agent(ConsoleDebugBrain())
    model = load_model(MODEL_FILE_BEST)
    network_agent = Agent(NetworkBrain(0.0, 50, lambda x: predict(model, x), lambda x: predict(model, x)))
    env = GameEnv(board, player, network_agent)
    r = env.play()
    print(r)

    #board = TicTacToeBoard()
    #self_play(MODEL_FILE_BEST, HISTORY_FOLDER, board, 1.0, 2)

    #dual_network()
    #self_play(MODEL_FILE_BEST)
    #model = load_model('./model/tictactoe/best.keras')

    #networkAgent = Agent(NetworkBrain(1.0, 50, lambda x: predict(model, x), lambda x: predict(model, x)))
    #randomAgent = Agent(MonteCarloBrain())
    #board = TicTacToeBoard()
    #env = GameEnv(board, networkAgent, randomAgent, lambda x: print(x), lambda x, y: print(x, y), lambda x, y: print(x, y))
    #env.play_n(5)

    #dual_network()
    #model = load_model('./model/tictactoe/best.keras')
    #history = play(model, TicTacToeBoard(), 1.0)

    # 学習データの保存
    #write_data(history)


    # モデルの破棄
    K.clear_session()
    del model

    #train_network()


    #train_network_tictactoe(1)
