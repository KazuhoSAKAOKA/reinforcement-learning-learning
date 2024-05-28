from typing import Callable, Tuple
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
import tensorflow as tf
import tensorflow.keras.initializers as initializers
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import os
from game_board import GameBoard, get_first_player_value
from gomoku_board import GomokuBoard
from mcts_node import pv_mcts_scores
from game import GameEnv, GameStats
from agent import Agent
from brains import RandomBrain, ConsoleDebugBrain
from mini_max import AlphaBetaBrain
from montecarlo import MonteCarloBrain
from network_common import judge_stats, train_network, self_play, train_cycle, train_cycle_dualmodel,evaluate_model
from network_brain import predict,NetworkBrain
from selfplay_brain import HistoryUpdater, ZeroToOneHistoryUpdater
from parameter import PARAM
from self_play import self_play_dualmodel, self_play
from selfplay_brain import SelfplayRandomBrain
from google_colab_helper import google_drive_path

DN_FILTERS = 128
DN_RESIDUAL_NUM = 16

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


def get_model_file_first_best_gcolab(board_size: int=15)->str:
  return google_drive_path + 'model/gomoku_{0}/first/best.keras'.format(board_size)
def get_history_folder_first_gcolab(board_size: int=15)->str:
  return google_drive_path + 'history/gomoku_{0}/first'.format(board_size)

def get_model_file_second_best_gcolab(board_size: int=15)->str:
  return google_drive_path + 'model/gomoku_{0}/second/best.keras'.format(board_size)
def get_history_folder_second_gcolab(board_size: int=15)->str:
  return google_drive_path + 'history/gomoku_{0}/second'.format(board_size)


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

def dual_network_1(file_best :str, board_size :int, show_summary:bool = False)->bool:
    if os.path.exists(file_best):
        return False
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
    if show_summary:
        model.summary()
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.save(file_best)

    del model
    tf.keras.backend.clear_session()
    return True


def dual_network(file_best :str, board_size :int, show_summary:bool = False)->bool:
    if os.path.exists(file_best):
        return False
    parent = os.path.dirname(file_best)
    os.makedirs(parent, exist_ok=True)
    
    input = Input(shape=(board_size, board_size, 2))

    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    x = GlobalAveragePooling2D()(x)

    p = Dense(board_size * board_size, kernel_initializer='he_normal', activation='relu')(x)
    p = Dense(board_size * board_size, kernel_regularizer=l2(0.0005), activation='softmax',name='pi')(p)

    v = Dense(board_size * board_size, kernel_initializer='he_normal', activation='relu')(x)
    v = Dense(1, kernel_regularizer=l2(0.0005), activation='tanh', name='v')(v)

    model = Model(inputs=input, outputs=[p,v])
    if show_summary:
        model.summary()
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.save(file_best)

    del model
    tf.keras.backend.clear_session()
    return True

def train_cycle_gomoku(
                board_size : int = 15,
                brain_evaluate_count : int = 50,
                selfplay_repeat : int = 500,
                epoch_count : int = 200,
                cycle_count : int = 10,
                eval_count: int = 20,
                eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats,
                use_cache: bool = True,
                new_model: bool = False,
                initial_selfplay_repeat: int = 1000,
                initial_train_count: int = 500,
                history_updater: HistoryUpdater=HistoryUpdater(),
                is_continue :bool = False,
                start_index:int = 0):  
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
        eval_judge=eval_judge,
        use_cache=use_cache,
        new_model=new_model,
        initial_selfplay_repeat=initial_selfplay_repeat,
        initial_train_count=initial_train_count,
        history_updater=history_updater,
        is_continue=is_continue,
        start_index=start_index)


def train_cycle_dualmodel_gomoku(
                board_size : int = 15,
                brain_evaluate_count : int = 50,
                selfplay_repeat : int = 500,
                epoch_count : int = 200,
                cycle_count : int = 10,
                eval_count: int = 20,
                eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats,
                use_cache: bool = True,
                initial_selfplay_repeat: int = 1000,
                initial_train_count: int = 500,
                history_updater: HistoryUpdater = ZeroToOneHistoryUpdater(),
                is_continue :bool = False,
                start_index:int = 0):  
    first_best_file = get_model_file_best_first(board_size)
    second_best_file = get_model_file_best_second(board_size)
    exist_first = dual_network(first_best_file,board_size)
    exist_second = dual_network(second_best_file,board_size)
    new_model = exist_first or exist_second

    train_cycle_dualmodel(
        game_board= GomokuBoard(board_size=board_size),
        brain_evaluate_count=brain_evaluate_count,
        first_best_model_file=first_best_file,
        second_best_model_file=second_best_file,
        history_first_folder=get_history_folder_first(board_size),
        history_second_folder=get_history_folder_second(board_size), 
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_judge=eval_judge,
        use_cache=use_cache,
        new_model=new_model,
        initial_selfplay_repeat=initial_selfplay_repeat,
        initial_train_count=initial_train_count,
        history_updater=history_updater,
        is_continue=is_continue,
        start_index=start_index)

def train_cycle_gomoku_gcolab(
                board_size : int = 15
                ,selfplay_repeat : int = 20
                ,epoch_count : int = 100
                ,cycle_count : int = 10
                ,eval_count: int = 0
                ,eval_temperature:float = 1.0
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats
                ,use_cache: bool = True):
    dual_network(get_model_file_best_gcolab(board_size),board_size)
    train_cycle(
        game_board= GomokuBoard(),
        best_model_file=get_model_file_best_gcolab(board_size),
        history_folder=get_history_folder_gcolab(board_size),
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_judge=eval_judge,
        use_cache=use_cache)


def train_cycle_dualmodel_gomoku_gcolab(
                board_size : int = 15
                ,brain_evaluate_count : int = 50
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats
                ,use_cache: bool = True):            
    first_best_file = get_model_file_first_best_gcolab(board_size)
    second_best_file = get_model_file_second_best_gcolab(board_size)
    dual_network(first_best_file,board_size)
    dual_network(second_best_file,board_size)
    train_cycle_dualmodel(
        game_board= GomokuBoard(board_size=board_size),
        brain_evaluate_count=brain_evaluate_count,
        first_best_model_file=first_best_file,
        second_best_model_file=second_best_file,
        history_first_folder=get_history_folder_first_gcolab(board_size),
        history_second_folder=get_history_folder_second_gcolab(board_size), 
        selfplay_repeat= selfplay_repeat,
        epoch_count= epoch_count ,
        cycle_count=cycle_count,
        eval_count=eval_count ,
        eval_judge=eval_judge,
        use_cache=use_cache)

