from typing import Callable, Tuple
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
from game import GameEnv, GameStats
from agent import Agent
from brains import RandomBrain, ConsoleDebugBrain
from mini_max import AlphaBetaBrain
from montecarlo import MonteCarloBrain
from network_common import judge_stats, train_network, self_play, train_cycle, train_2_cycle,evaluate_model
from network_brain import predict,DualModelNetworkBrain,NetworkBrain
from parameter import PARAM
from self_play import write_data, load_data, load_data_file, load_data_file_name
from google_colab_helper import google_drive_path

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


def get_model_file_best_gcolab()->str:
  return google_drive_path + 'model/tictactoe/best.keras'
def get_history_folder_gcolab()->str:
  return google_drive_path + 'history/tictactoe'

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

def dual_network_1(file_best=MODEL_FILE_BEST):
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
    print(model.summary())

    model.save(file_best)

    K.clear_session()
    del model
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

    p = conv(DN_FILTERS)(x)
    p = BatchNormalization()(p)
    p = Activation('relu')(p)
    p = GlobalAveragePooling2D()(p)
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005), activation='softmax',name='pi')(p)

    v = conv(DN_FILTERS)(x)
    v = BatchNormalization()(v)
    v = Activation('relu')(v)
    v = GlobalAveragePooling2D()(v)
    v = Dense(1, kernel_regularizer=l2(0.0005))(v)
    v = Activation('tanh', name='v')(v)

    model = Model(inputs=input, outputs=[p,v])
    model.summary()
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])

    model.save(file_best)

    K.clear_session()
    del model


RN_EPOCHS=200
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

def tictac_selfplay2():
    dual_network(MODEL_FILE_BEST_FIRST)
    dual_network(MODEL_FILE_BEST_SECOND)

    train_network_tictactoe()

def tictac_train_first():
    train_network(MODEL_FILE_BEST_FIRST, HISTORY_FOLDER_FIRST, TicTacToeBoard(), RN_EPOCHS)

def tictac_train_second():
    train_network(MODEL_FILE_BEST_SECOND, HISTORY_FOLDER_SECOND, TicTacToeBoard(), RN_EPOCHS)


def evaluate():
    model = load_model('./model/tictactoe/20240413091938.keras')
    network_agent = Agent(NetworkBrain(0.0, 200, model))
    random_agent = Agent(RandomBrain())
    alpha_beta_agent = Agent(AlphaBetaBrain())
    montecarlo_agent = Agent(MonteCarloBrain(50))

    r1,r2 = evaluate_model(network_agent, random_agent, TicTacToeBoard(), 20)
    print(r1)
    print(r2)
    r1,r2 = evaluate_model(network_agent, alpha_beta_agent, TicTacToeBoard(), 20)
    print(r1)
    print(r2)
    r1,r2 = evaluate_model(network_agent, montecarlo_agent, TicTacToeBoard(), 20)
    print(r1)
    print(r2)

    # モデルの破棄
    K.clear_session()
    del model
def evaluate2():
    first_model = load_model('./model/tictactoe/first/20240412054736.keras')
    second_model = load_model('./model/tictactoe/second/20240412114427.keras')
    network_agent = Agent(DualModelNetworkBrain(0.0, 200, first_model, second_model))
    random_agent = Agent(RandomBrain())
    alpha_beta_agent = Agent(AlphaBetaBrain())
    montecarlo_agent = Agent(MonteCarloBrain(50))

    r1,r2 = evaluate_model(network_agent, random_agent, TicTacToeBoard(), 20)
    print(r1)
    print(r2)
    r1,r2 = evaluate_model(network_agent, alpha_beta_agent, TicTacToeBoard(), 20)
    print(r1)
    print(r2)
    r1,r2 = evaluate_model(network_agent, montecarlo_agent, TicTacToeBoard(), 20)
    print(r1)
    print(r2)

    # モデルの破棄
    K.clear_session()
    del first_model
    del second_model


def train_cycle_tictactoe(
                brain_evaluate_count : int = 50
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats):        
    dual_network(MODEL_FILE_BEST)
    train_cycle(
        game_board= TicTacToeBoard(),
        brain_evaluate_count= brain_evaluate_count,
        best_model_file=MODEL_FILE_BEST, 
        history_folder=HISTORY_FOLDER,
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_judge=eval_judge)


def train2_cycle_tictactoe(selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_temperature:float = 1.0
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats):
    dual_network(MODEL_FILE_BEST_FIRST)
    dual_network(MODEL_FILE_BEST_SECOND)
    train_2_cycle(
        first_best_model_file=MODEL_FILE_BEST_FIRST,
        second_best_model_file=MODEL_FILE_BEST_SECOND,
        history_first_folder=HISTORY_FOLDER_FIRST,
        history_second_folder=HISTORY_FOLDER_SECOND, 
        game_board= TicTacToeBoard(),
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_temperature=eval_temperature,
        eval_judge=eval_judge)

def train_cycle_tictactoe_gcolab(
                brain_evaluate_count : int = 50
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats):             
    dual_network(MODEL_FILE_BEST)
    train_cycle(
        game_board= TicTacToeBoard(),
        brain_evaluate_count= brain_evaluate_count,
        best_model_file=MODEL_FILE_BEST, 
        history_folder=HISTORY_FOLDER,
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_judge=eval_judge)


if __name__ == '__main__':

    evaluate2()
    #dual_network(MODEL_FILE_BEST_FIRST)
    #dual_network(MODEL_FILE_BEST_SECOND)
    #train_network(MODEL_FILE_BEST_FIRST, HISTORY_FOLDER_FIRST, TicTacToeBoard(), 20000)
    #train_network(MODEL_FILE_BEST_SECOND, HISTORY_FOLDER_SECOND, TicTacToeBoard(), 20000)
    #dual_network(MODEL_FILE_BEST)
    #train_network(MODEL_FILE_BEST, HISTORY_FOLDER, TicTacToeBoard(), 20000)
    #train2()
    #dual_network(MODEL_FILE_BEST)
    #train_network(MODEL_FILE_BEST, HISTORY_FOLDER_FIRST, TicTacToeBoard(), RN_EPOCHS)
    #tictac_selfplay2()
    #tictac_train_first()
    #tictac_train_second()
