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
from game import GameEnv, GameStats
from agent import Agent
from brains import RandomBrain, ConsoleDebugBrain
from mini_max import AlphaBetaBrain
from montecarlo import MonteCarloBrain
from network_common import judge_stats, train_network, train_cycle, evaluate_model
from network_brain import NetworkBrain,NetworkBrainFactory
from google_colab_helper import google_drive_path
from parameter import NetworkParameter,SelfplayParameter,BrainParameter,ExplorationParameter,NetworkType,HistoryUpdateType,ActionSelectorType, InitSelfplayParameter

DN_FILTERS = 128
DN_RESIDUAL_NUM = 16
DN_INPUT_SHAPE = (3,3,2)
DN_OUTPUT_SIZE = 9

MODEL_FOLDER = './model/tictactoe'
HISTORY_FOLDER = './data/tictactoe'

def get_model_folder_gcolab()->str:
  return google_drive_path + 'model/tictactoe'
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
# AlphaZeroの書籍に乗ってたネットワーク
def dual_network_past()->Model:
   
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
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])
    return model

def create_dual_network_model()->Model:
    
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
    
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])
    return model

def convert(xs):
    a,b,c = (3,3,2)
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)

    return xs

TICTACTOE_NETWORK_PARAM = NetworkParameter(
        model_folder=MODEL_FOLDER, 
        network_type=NetworkType.DualNetwork,
        is_dual_model=False)
TICTACTOE_NETWORK_PARAM_DUAL = NetworkParameter(
        model_folder=MODEL_FOLDER, 
        network_type=NetworkType.DualNetwork,
        is_dual_model=True)
TICTACTOE_NETWORK_PARAM_GCOLAB = NetworkParameter(
        model_folder=get_model_folder_gcolab(), 
        network_type=NetworkType.DualNetwork,
        is_dual_model=False)
TICTACTOE_NETWORK_PARAM_DUAL_GCOLAB = NetworkParameter(
        model_folder=get_model_folder_gcolab(), 
        network_type=NetworkType.DualNetwork,
        is_dual_model=True)



TICTACTOE_SELFPLAY_PARAM = SelfplayParameter(
    history_folder=HISTORY_FOLDER,
    cycle_count=10,
    selfplay_repeat=500,
    evaluate_count=50,
    continue_history_folder_path = None)
TICTACTOE_SELFPLAY_PARAM_DUAL = SelfplayParameter(
    history_folder=HISTORY_FOLDER,
    cycle_count=10,
    selfplay_repeat=500,
    evaluate_count=50,
    continue_history_folder_path = None)

TICTACTOE_SELFPLAY_PARAM_GCOLAB = SelfplayParameter(
    history_folder=get_history_folder_gcolab(),
    cycle_count=10,
    selfplay_repeat=500,
    evaluate_count=50,
    continue_history_folder_path = None)
TICTACTOE_SELFPLAY_PARAM_DUAL_GCOLAB = SelfplayParameter(
    history_folder=get_history_folder_gcolab(),
    cycle_count=10,
    selfplay_repeat=500,
    evaluate_count=50,
    continue_history_folder_path = None)


TICTACTOE_BRAIN_PARAM = BrainParameter(
    mcts_evaluate_count=50,
    mcts_expand_limit=10,
    use_cache=True,
    history_update_type=HistoryUpdateType.zero_to_one,
    action_selector_type=ActionSelectorType.max)

TICTACTOE_INIT_TRAIN_PARAM=InitSelfplayParameter(
    selfplay_repeat=1500,
    train_epoch=500)

def train_cycle_tictactoe(
                board_size=3,
                network_param: NetworkParameter = TICTACTOE_NETWORK_PARAM,
                selfplay_param: SelfplayParameter=TICTACTOE_SELFPLAY_PARAM,
                brain_param: BrainParameter=TICTACTOE_BRAIN_PARAM,
                exploration_param: ExplorationParameter=ExplorationParameter(),
                initial_selfplay_param: SelfplayParameter = TICTACTOE_INIT_TRAIN_PARAM):

    train_cycle(
        game_board= TicTacToeBoard(),
        create_model=create_dual_network_model,
        network_param=network_param,
        selfplay_param=selfplay_param,
        brain_param=brain_param,
        exploration_param=exploration_param,
        initial_selfplay_param=initial_selfplay_param)

def train_cycle_tictactoe_dual(
                board_size=3,
                network_param: NetworkParameter = TICTACTOE_NETWORK_PARAM_DUAL,
                selfplay_param: SelfplayParameter=TICTACTOE_SELFPLAY_PARAM_DUAL,
                brain_param: BrainParameter=TICTACTOE_BRAIN_PARAM,
                exploration_param: ExplorationParameter=ExplorationParameter(),
                initial_selfplay_param: SelfplayParameter = TICTACTOE_INIT_TRAIN_PARAM):

    train_cycle(
        game_board= TicTacToeBoard(),
        create_model=create_dual_network_model,
        network_param=network_param,
        selfplay_param=selfplay_param,
        brain_param=brain_param,
        exploration_param=exploration_param,
        initial_selfplay_param=initial_selfplay_param)


def train_cycle_tictactoe_gcolab(
                board_size=3,
                network_param: NetworkParameter = TICTACTOE_NETWORK_PARAM_GCOLAB,
                selfplay_param: SelfplayParameter=TICTACTOE_SELFPLAY_PARAM_GCOLAB,
                brain_param: BrainParameter=TICTACTOE_BRAIN_PARAM,
                exploration_param: ExplorationParameter=ExplorationParameter(),
                initial_selfplay_param: SelfplayParameter = TICTACTOE_INIT_TRAIN_PARAM):

    train_cycle(
        game_board= TicTacToeBoard(),
        create_model=create_dual_network_model,
        network_param=network_param,
        selfplay_param=selfplay_param,
        brain_param=brain_param,
        exploration_param=exploration_param,
        initial_selfplay_param=initial_selfplay_param)

def train_cycle_tictactoe_dual_gcolab(
                board_size=3,
                network_param: NetworkParameter = TICTACTOE_NETWORK_PARAM_DUAL_GCOLAB,
                selfplay_param: SelfplayParameter=TICTACTOE_SELFPLAY_PARAM_DUAL_GCOLAB,
                brain_param: BrainParameter=TICTACTOE_BRAIN_PARAM,
                exploration_param: ExplorationParameter=ExplorationParameter(),
                initial_selfplay_param: SelfplayParameter = TICTACTOE_INIT_TRAIN_PARAM):

    train_cycle(
        game_board= TicTacToeBoard(),
        create_model=create_dual_network_model,
        network_param=network_param,
        selfplay_param=selfplay_param,
        brain_param=brain_param,
        exploration_param=exploration_param,
        initial_selfplay_param=initial_selfplay_param)
