import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from datetime import datetime
from self_play import self_play, self_play_dualmodel, load_data_file_name
import numpy as np
import pickle
import os
import shutil
import concurrent.futures

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

def train_network(load_model_path, history_file, game_board : GameBoard, epoch_count : int)->Tuple[Model, str]:

    model = tf.keras.models.load_model(load_model_path)
    # model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    history = load_data_file_name(history_file)
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
    save_path = os.path.dirname(load_model_path) + '/{:04}{:02}{:02}{:02}{:02}{:02}.keras'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    
    #f.saved_model.save(model, save_path)
    model.save(save_path)
    return model, save_path


def evaluate_model(agent_target : Agent , agent_base: Agent, board : GameBoard, play_count: int, executor:concurrent.futures.ThreadPoolExecutor = None)->Tuple[GameStats, GameStats]:

    if executor is None:
        env = GameEnv(board, agent_target, agent_base, episode_callback=lambda i, _1, _2: print('\rEvaluate {}/{}'.format(i + 1, play_count), end=''))
        first = env.play_n(play_count)

        env = GameEnv(board, agent_base, agent_target, episode_callback=lambda i, _1, _2: print('\rEvaluate {}/{}'.format(i + 1, play_count), end=''))
        second = env.play_n(play_count)

        return first, second
    else:
        def execute(agent_target, agent_base, board, play_count):
            env = GameEnv(board, agent_target, agent_base)
            return env.play_n(play_count)

        future_first = executor.submit(lambda: execute(agent_target, agent_base, board, play_count))
        future_second = executor.submit(lambda: execute(agent_base, agent_target, board, play_count))
        return future_first.result(), future_second.result()


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
    if eval_count > 0:
        executor = concurrent.futures.ThreadPoolExecutor(2)
    else:
        executor = None

    for i in range(cycle_count):
        print('cycle {}/{}'.format(i + 1, cycle_count))
        
        best_model = tf.keras.models.load_model(best_model_file)
        first_brain = SelfplayNetworkBrain(brain_evaluate_count, best_model)
        second_brain = SelfplayNetworkBrain(brain_evaluate_count, best_model)
        history_file = self_play(first_brain, second_brain, game_board, selfplay_repeat, history_folder)
        latest_model, latest_file_name = train_network(best_model_file, history_file, game_board, epoch_count)

        print('training latest model file={0}'.format(latest_file_name))
        replace = True
        if eval_count > 0:
            latest_brain = NetworkBrain(brain_evaluate_count, latest_model)
            best_brain = NetworkBrain(brain_evaluate_count, best_model)
            stats = evaluate_model(agent_target=Agent(brain=latest_brain, name='latest'), agent_base=Agent(brain=best_brain, name='best'), board=game_board,play_count=eval_count, executor=executor)
            replace = eval_judge(stats)
        if replace:
            os.remove(best_model_file)
            shutil.copy(latest_file_name, best_model_file)
            print("latest model replace best model")
        del best_model
        del latest_model
        tf.keras.backend.clear_session()

def train_cycle_dualmodel(game_board : GameBoard
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
    executor = concurrent.futures.ThreadPoolExecutor(2)
    for i in range(cycle_count):
        print('cycle {}/{}'.format(i + 1, cycle_count))
        first_model = tf.keras.models.load_model(first_best_model_file)
        second_model = tf.keras.models.load_model(second_best_model_file)
        first_brain = SelfplayDualModelNetworkBrain(brain_evaluate_count, first_model, second_model)
        second_brain = SelfplayDualModelNetworkBrain(brain_evaluate_count, first_model, second_model)
        first_history_file, second_history_fine = self_play_dualmodel(first_brain, second_brain,game_board, selfplay_repeat, history_first_folder, history_second_folder)
        
        future_first = executor.submit(lambda: train_network(first_best_model_file, first_history_file, game_board, epoch_count))
        future_second = executor.submit(lambda: train_network(second_best_model_file, second_history_fine, game_board, epoch_count))
        latest_first_model, latest_file_name_first = future_first.result()
        latest_second_model, latest_file_name_second = future_second.result()
        # latest_file_name_first = train_network(first_best_model_file, first_history_file, game_board, epoch_count)
        # latest_file_name_second = train_network(second_best_model_file, second_history_fine, game_board, epoch_count)
        print('training first model file={0}, second model file={1}'.format(latest_file_name_first, latest_file_name_second))
        # latest_first_model = tf.saved_model.load(latest_file_name_first)
        # latest_second_model = tf.saved_model.load(latest_file_name_second)
        #best_first_model = tf.saved_model.load(first_best_model_file)    
        #best_second_model = tf.saved_model.load(second_best_model_file)

        replace = True
        if eval_count > 0:
            latest_brain = DualModelNetworkBrain(brain_evaluate_count, latest_first_model, latest_second_model)
            best_brain = DualModelNetworkBrain(brain_evaluate_count, first_model, second_model)            
            stats = evaluate_model(agent_target=Agent(brain=latest_brain, name='latest'), agent_base=Agent(brain=best_brain, name='best'), board=game_board,play_count=eval_count, executor=executor)
            replace = eval_judge(stats)
        if replace:
            os.remove(first_best_model_file)
            shutil.copy(latest_file_name_first, first_best_model_file)
            print("first model replace best model")

            os.remove(second_best_model_file)
            shutil.copy(latest_file_name_second, second_best_model_file)
            print("second model replace best model") 
        del latest_first_model
        del latest_second_model
        del second_model
        del first_model
        tf.keras.backend.clear_session()
