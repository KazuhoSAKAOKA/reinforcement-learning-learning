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
from game import GameEnv, GameStats
from agent import Agent
from brains import RandomBrain, ConsoleDebugBrain
from mini_max import AlphaBetaBrain
from montecarlo import MonteCarloBrain
from network_common import train_cycle
from self_play_brain import HistoryUpdater, ZeroToOneHistoryUpdater
from parameter import NetworkParameter,SelfplayParameter,BrainParameter,ExplorationParameter,NetworkType,HistoryUpdateType,ActionSelectorType, InitSelfplayParameter, judge_stats
from self_play import self_play_impl
from self_play_brain import SelfplayRandomBrain
from google_colab_helper import google_drive_path

DN_FILTERS = 128
DN_RESIDUAL_NUM = 16


def get_model_folder(board_size : int = 15)->str:
    return './model/gomoku_{0}'.format(board_size)

def get_history_folder(board_size : int = 15)->str:
    return './data/gomoku_{0}'.format(board_size)

def get_model_folder_gcolab(board_size: int=15)->str:
    return google_drive_path + 'model/gomoku_{0}'.format(board_size)

def get_history_folder_gcolab(board_size: int=15)->str:
    return google_drive_path + 'model/gomoku_{0}'.format(board_size)


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

def create_dual_network(board_size :int)->Model:
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
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model


def dual_network_A(file_best :str, board_size :int, show_summary:bool = False)->bool:
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

def create_network_parameter(board_size : int = 15)->NetworkParameter:
    return NetworkParameter(
        model_folder=get_model_folder(board_size=board_size), 
        network_type=NetworkType.DualNetwork,
        is_dual_model=False)
def create_network_parameter_dual(board_size : int = 15)->NetworkParameter:
    return NetworkParameter(
        model_folder=get_model_folder(board_size=board_size), 
        network_type=NetworkType.DualNetwork,
        is_dual_model=True)

def create_selfplay_parameter(board_size : int = 15)->SelfplayParameter:
    return SelfplayParameter(
        history_folder=get_history_folder(board_size=board_size),
        cycle_count=10,
        selfplay_repeat=100,
        evaluate_count=20,
        eval_judge=judge_stats,
        train_epoch=200,
        continue_history_folder_path=None)

def create_brain_parameter(board_size : int = 15)->BrainParameter:
    return BrainParameter(
        mcts_evaluate_count=50,
        mcts_expand_limit=10,
        use_cache=True,
        history_update_type=HistoryUpdateType.zero_to_one,
        action_selector_type=ActionSelectorType.max)

def create_initial_selfplay_parameter(board_size : int = 15)->InitSelfplayParameter:
    return InitSelfplayParameter(
        selfplay_repeat=100,
        train_epoch=200)

def train_cycle_gomoku(
                board_size:int,
                network_param: NetworkParameter,
                selfplay_param: SelfplayParameter,
                brain_param: BrainParameter,
                exploration_param: ExplorationParameter,
                initial_selfplay_param: SelfplayParameter):

    train_cycle(
        game_board= GomokuBoard(board_size=board_size),
        create_model=lambda :create_dual_network(board_size),
        network_param=network_param,
        selfplay_param=selfplay_param,
        brain_param=brain_param,
        exploration_param=exploration_param,
        initial_selfplay_param=initial_selfplay_param)

def train_cycle_gomoku_default(board_size=15):
    pass
def train_cycle_gomoku_dual_default(board_size=15):
    pass
