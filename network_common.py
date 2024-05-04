from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from datetime import datetime
from self_play import self_play, self_play2, SelfplayBrain, load_data, load_data_file, load_data_file_name
import numpy as np
import pickle
import os
import shutil

from game_board import GameBoard, get_first_player_value
from pv_mcts import pv_mcts_scores
from game import GameEnv, GameStats
from agent import Agent
from brains import RandomBrain
from montecarlo import MonteCarloBrain
from typing import Callable
from parameter import PARAM
from typing import Tuple
from selfplay_brain import SelfplayBrain
from network_brain import NetworkBrain, SelfplayBrain, SelfplayNetworkBrain, SelfplayDualModelNetworkBrain, DualModelNetworkBrain



def self_play_model(model_file : str, history_folder : str, board : GameBoard, temperature: float, repeat_count: int)->str:
    model = load_model(model_file)
    first_brain = SelfplayNetworkBrain(temperature, PARAM.evaluate_count, model)
    second_brain = SelfplayNetworkBrain(temperature, PARAM.evaluate_count, model)
    history_file = self_play(first_brain, second_brain, board, repeat_count, history_folder)
    K.clear_session()
    del model
    return history_file

def self_play2_model(first_model_file : str, second_model_file: str, history_first_folder : str, history_second_folder : str, board : GameBoard, temperature:float, repeat_count:int)->Tuple[str,str]:
    first_model = load_model(first_model_file)
    second_model = load_model(second_model_file)
    first_brain = SelfplayNetworkBrain(temperature, PARAM.evaluate_count, first_model, second_model)
    second_brain = SelfplayNetworkBrain(temperature, PARAM.evaluate_count, first_model, second_model)
    history_files = self_play(first_brain, second_brain, board, repeat_count, history_first_folder, history_second_folder)
    K.clear_session()
    del first_model
    del second_model
    return history_files
    
def train_network(load_model_path, history_folder, game_board : GameBoard, epoch_count : int):

    model = load_model(load_model_path)
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    history = load_data(history_folder)
    xs, y_policies, y_values = game_board.reshape_history_to_input(history)

    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    print_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: print('\rTrain {}/{}'.format(epoch + 1, epoch_count), end=''))

    model.fit(xs, [y_policies, y_values], batch_size=128, epochs=epoch_count, 
              verbose=0, callbacks=[lr_decay, print_callback])

    print('\rcomplete. train')
    now = datetime.now()
    save_file_name = os.path.dirname(load_model_path) + '/{:04}{:02}{:02}{:02}{:02}{:02}.keras'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    model.save(save_file_name)

    K.clear_session()
    del model
    return save_file_name


def evaluate_model(agent_target : Agent , agent_base: Agent, board : GameBoard, play_count: int)->Tuple[GameStats, GameStats]:

    env = GameEnv(board, agent_target, agent_base, episode_callback=lambda i, _1, _2: print('\rEvaluate {}/{}'.format(i + 1, play_count), end=''))
    first = env.play_n(play_count)

    env = GameEnv(board, agent_base, agent_target, episode_callback=lambda i, _1, _2: print('\rEvaluate {}/{}'.format(i + 1, play_count), end=''))
    second = env.play_n(play_count)

    return first, second



def judge_stats(stats: Tuple[GameStats, GameStats])->bool:
    current, best = stats
    if current.first_player_win > best.first_player_win:
        return True
    if current.first_player_win == best.first_player_win and current.draw > best.draw:
        return True
    return False


def train_cycle(
                game_board : GameBoard
                ,best_model_file : str
                ,history_folder : str
                ,brain_evaluate_count : int = 100
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats):
    for i in range(cycle_count):
        print('cycle {}/{}'.format(i + 1, cycle_count))
        
        model = load_model(best_model_file)
        first_brain = SelfplayNetworkBrain(brain_evaluate_count, model)
        second_brain = SelfplayNetworkBrain(brain_evaluate_count, model)
        self_play(first_brain, second_brain, game_board, selfplay_repeat, history_folder)
        latest_file_name = train_network(best_model_file, history_folder, game_board, epoch_count)
        best_model = load_model(best_model_file)
        latest_model = load_model(latest_file_name)

        print('training latest model file={0}'.format(latest_file_name))
        replace = True
        if eval_count > 0:
            latest_brain = NetworkBrain(brain_evaluate_count, latest_model)
            best_brain = NetworkBrain(brain_evaluate_count, best_model)
            stats = evaluate_model(agent_target=Agent(brain=latest_brain, name='latest'), agent_base=Agent(brain=best_brain, name='best'), board=game_board,play_count=eval_count)
            replace = eval_judge(stats)
        if replace:
            os.remove(best_model_file)
            shutil.copy(latest_file_name, best_model_file)
            print("latest model replace best model")

def train_2_cycle(game_board : GameBoard
                ,brain_evaluate_count : int
                ,first_best_model_file : str
                ,second_best_model_file : str
                ,history_first_folder : str
                ,history_second_folder : str
                ,selfplay_repeat : int = 500
                ,epoch_count : int = 200
                ,cycle_count : int = 10
                ,eval_count: int = 20
                ,eval_temperature:float = 1.0
                ,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats):
    for i in range(cycle_count):
        print('cycle {}/{}'.format(i + 1, cycle_count))
        first_model = load_model(first_best_model_file)
        second_model = load_model(second_best_model_file)
        first_brain = SelfplayDualModelNetworkBrain(brain_evaluate_count, first_model, second_model)
        second_brain = SelfplayDualModelNetworkBrain(brain_evaluate_count, first_model, second_model)
        self_play2(first_brain, second_brain,game_board, selfplay_repeat, history_first_folder, history_second_folder)
        
        latest_file_name_first = train_network(first_best_model_file, history_first_folder, game_board, epoch_count)
        latest_file_name_second = train_network(second_best_model_file, history_second_folder, game_board, epoch_count)
        print('training first model file={0}, second model file={1}'.format(latest_file_name_first, latest_file_name_second))
        latest_first_model = load_model(latest_file_name_first)
        latest_second_model = load_model(latest_file_name_second)
        best_first_model = load_model(first_best_model_file)    
        best_second_model = load_model(second_best_model_file)

        replace = True
        if eval_count > 0:
            latest_brain = DualModelNetworkBrain(eval_temperature, brain_evaluate_count, latest_first_model, latest_second_model)
            best_brain = DualModelNetworkBrain(eval_temperature, brain_evaluate_count, best_first_model, best_second_model)            
            stats = evaluate_model(Agent(latest_brain), Agent(best_brain), game_board, eval_count)
            replace = eval_judge(stats)
        if replace:
            os.remove(first_best_model_file)
            shutil.copy(latest_file_name_first, first_best_model_file)
            print("first model replace best model")

            os.remove(second_best_model_file)
            shutil.copy(latest_file_name_second, second_best_model_file)
            print("second model replace best model") 
        K.clear_session()
        del latest_first_model
        del latest_second_model
        del best_second_model
        del best_first_model

