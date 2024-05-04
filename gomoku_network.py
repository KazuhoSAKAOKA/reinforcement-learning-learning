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
from gomoku_board import GomokuBoard
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

#MODEL_FILE_BEST = './model/gomoku/best.keras'
#MODEL_FILE_LATEST = './model/gomoku/latest.keras'
#HISTORY_FOLDER = './data/gomoku'
def get_history_folder(board_size : int = 15)->str:
    return './data/gomoku_{0}'.format(board_size)

def get_model_file_best(board_size : int = 15)->str:
    return './model/gomoku_{0}/best.keras'.format(board_size)

def get_history_folder_first(board_size : int = 15)->str:
    return './data/gomoku_{0}/first'.format(board_size)
def get_history_folder_second(board_size : int = 15)->str:
    return './data/gomoku_{0}/second'.format(board_size)
def get_model_file_best_first(board_size : int = 15)->str:
    return './model/gomoku_{0}/first/best.keras'.format(board_size)
def get_model_file_best_second(board_size : int = 15)->str:
    return './model/gomoku_{0}/second/best.keras'.format(board_size)

def get_model_file_best_gcolab(board_size: int=15)->str:
  return google_drive_path + 'model/gomoku_{0}/best.keras'.format(board_size)
def get_history_folder_gcolab(board_size: int=15)->str:
  return google_drive_path + 'history/gomoku_{0}'.format(board_size)

def conv(filters):
    return Conv2D(filters, 5, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

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

def dual_network(file_best :str, board_size :int):
    if os.path.exists(file_best):
        return
    parent = os.path.dirname(file_best)
    os.makedirs(parent, exist_ok=True)
    
    input = Input(shape=(board_size, board_size, 2))

    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    p = conv(DN_FILTERS)(x)
    p = BatchNormalization()(p)
    p = Activation('relu')(p)
    p = GlobalAveragePooling2D()(p)
    p = Dense(board_size * board_size, kernel_regularizer=l2(0.0005), activation='softmax',name='pi')(p)

    v = conv(DN_FILTERS)(x)
    v = BatchNormalization()(v)
    v = Activation('relu')(v)
    v = GlobalAveragePooling2D()(v)
    v = Dense(1, kernel_regularizer=l2(0.0005))(v)
    v = Activation('tanh', name='v')(v)

    model = Model(inputs=input, outputs=[p,v])
    model.summary()
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')
    model.save(file_best)

    K.clear_session()
    del model




def train_cycle_gomoku(
                board_size : int = 15
                ,brain_evaluate_count : int = 50
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats):     
    dual_network(get_model_file_best(board_size),board_size)
    train_cycle(
        game_board= GomokuBoard(board_size=board_size),
        brain_evaluate_count=brain_evaluate_count,
        best_model_file=get_model_file_best(board_size=board_size), 
        history_folder=get_history_folder(board_size=board_size),
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_judge=eval_judge)


def train2_cycle_gomoku(
                board_size : int = 15
                ,brain_evaluate_count : int = 50
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats):            
        
    dual_network(get_model_file_best_first(board_size),board_size)
    dual_network(get_model_file_best_second(board_size),board_size)
    train_2_cycle(
        first_best_model_file=get_model_file_best_first(board_size),
        second_best_model_file=get_model_file_best_second(board_size),
        history_first_folder=get_history_folder_first(board_size),
        history_second_folder=get_history_folder_second(board_size), 
        game_board= GomokuBoard(board_size=board_size),
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_temperature=eval_temperature,
        eval_judge=eval_judge)


def train_cycle_gomoku_gcolab(board_size : int = 15
                ,selfplay_repeat : int = 20
                ,epoch_count : int = 100
                ,cycle_count : int = 10
                ,eval_count: int = 0
                ,eval_temperature:float = 1.0
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats):
    dual_network(get_model_file_best_gcolab(board_size),board_size)
    train_cycle(
        best_model_file=get_model_file_best_gcolab(board_size),
        history_folder=get_history_folder_gcolab(board_size),
        game_board= GomokuBoard(),
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_temperature=eval_temperature,
        eval_judge=eval_judge)


if __name__ == '__main__':
    train_cycle_gomoku(15, 1, 1, 2, 0)

