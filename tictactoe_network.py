from typing import Callable, Tuple
import tensorflow as tf
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
from network_common import judge_stats, train_network, train_cycle, train_cycle_dualmodel, evaluate_model
from network_brain import predict,DualModelNetworkBrain,NetworkBrain
from parameter import PARAM
from self_play import write_data, load_data, load_data_file, load_data_file_name
from google_colab_helper import google_drive_path

DN_FILTERS = 128
DN_RESIDUAL_NUM = 16
DN_INPUT_SHAPE = (3,3,2)
DN_OUTPUT_SIZE = 9

MODEL_FILE_BEST = './model/tictactoe/best.keras'
HISTORY_FOLDER = './data/tictactoe'

MODEL_FILE_BEST_FIRST = './model/tictactoe/first/best.keras'
MODEL_FILE_BEST_SECOND = './model/tictactoe/second/best.keras'
HISTORY_FOLDER_FIRST = './data/tictactoe/first'
HISTORY_FOLDER_SECOND = './data/tictactoe/second'

def get_model_file_best_gcolab()->str:
  return google_drive_path + 'model/tictactoe/best.keras'
def get_history_folder_gcolab()->str:
  return google_drive_path + 'history/tictactoe'


def get_model_file_first_best_gcolab()->str:
  return google_drive_path + 'model/tictactoe/first/best.keras'
def get_model_file_second_best_gcolab()->str:
  return google_drive_path + 'model/tictactoe/second/best.keras'

def get_history_folder_first_gcolab()->str:
  return google_drive_path + 'history/tictactoe/first'
def get_history_folder_second_gcolab()->str:
  return google_drive_path + 'history/tictactoe/second'


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
# AlphaZeroの書籍に乗ってたネットワーク
def dual_network_past(file_best=MODEL_FILE_BEST):
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

def dual_network(path_best=MODEL_FILE_BEST, output_summary: bool = False):
    if os.path.exists(path_best):
        return
    parent = os.path.dirname(path_best)
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
    if output_summary:
        model.summary()
    
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])
    #tf.saved_model.save(model, path_best)
    model.save(path_best)

    K.clear_session()
    del model

def load_data():
    history_path = sorted(Path('./data/tictactoe').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

def convert(xs):
    a,b,c = (3,3,2)
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)

    return xs


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



def train_cycle_tictactoe(
                brain_evaluate_count : int = 50
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats
                ,use_cache = True):        
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
        eval_judge=eval_judge,
        use_cache=use_cache)


def train_cycle_dualmodel_tictactoe(
                brain_evaluate_count : int = 50
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats
                ,use_cache = True):
    dual_network(MODEL_FILE_BEST_FIRST)
    dual_network(MODEL_FILE_BEST_SECOND)
    train_cycle_dualmodel(
        game_board= TicTacToeBoard(),
        brain_evaluate_count= brain_evaluate_count,
        first_best_model_file=MODEL_FILE_BEST_FIRST,
        second_best_model_file=MODEL_FILE_BEST_SECOND,
        history_first_folder=HISTORY_FOLDER_FIRST,
        history_second_folder=HISTORY_FOLDER_SECOND, 
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_judge=eval_judge,
        use_cache=use_cache)

def train_cycle_tictactoe_gcolab(
                brain_evaluate_count : int = 50
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats
                ,use_cache = True):
    best_file = get_model_file_best_gcolab()         
    dual_network(best_file)
    train_cycle(
        game_board= TicTacToeBoard(),
        brain_evaluate_count= brain_evaluate_count,
        best_model_file=best_file, 
        history_folder=get_history_folder_gcolab(),
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_judge=eval_judge,
        use_cache=use_cache)


def train_cycle_dualmodel_tictactoe_gcolab(
                brain_evaluate_count : int = 50
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats
                ,use_cache = True):
    first_best_file = get_model_file_first_best_gcolab()         
    second_best_file = get_model_file_second_best_gcolab()         

    dual_network(first_best_file)
    dual_network(second_best_file)
    train_cycle_dualmodel(
        game_board= TicTacToeBoard(),
        brain_evaluate_count= brain_evaluate_count,
        first_best_model_file=first_best_file,
        second_best_model_file=second_best_file,
        history_first_folder=get_history_folder_first_gcolab(),
        history_second_folder=get_history_folder_second_gcolab, 
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_judge=eval_judge,
        use_cache=use_cache)