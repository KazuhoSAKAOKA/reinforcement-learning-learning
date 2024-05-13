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
from selfplay_brain import SelfplayRandomBrain, HistoryUpdater
from network_brain import NetworkBrain, SelfplayNetworkBrain, NetworkBrainFactory
from threadsafe_dict import ThreadSafeDict




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
    print('begin evaluate best vs latest')
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

def initial_train(
            game_board: GameBoard,
            best_model_file: str,
            history_folder: str,
            initial_selfplay_repeat: int,
            initial_train_count: int,
            history_updater: HistoryUpdater):
    print('initial selfplay and train')

    history_file = self_play(
        first_brain=SelfplayRandomBrain(history_updater=history_updater),
        second_brain= SelfplayRandomBrain(history_updater=history_updater),
        board=game_board,
        repeat_count=initial_selfplay_repeat,
        history_folder=history_folder)
    print('initial selfplay completed. begin train')
    model, latest_file_name = train_network(best_model_file, history_file, game_board, initial_train_count)

    print('initial training complete. model file={}'.format(latest_file_name))
    os.remove(best_model_file)
    shutil.copy(latest_file_name, best_model_file)
    del model
    tf.keras.backend.clear_session()


def train_cycle(
                game_board : GameBoard,
                best_model_file : str,
                history_folder : str,
                brain_evaluate_count : int = 100,
                selfplay_repeat : int = 500,
                epoch_count : int = 200,
                cycle_count : int = 10,
                eval_count: int = 20,eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats,
                use_cache = True,
                new_model: bool = False,
                initial_selfplay_repeat: int = 1000,
                initial_train_count: int = 500,
                history_updater: HistoryUpdater=HistoryUpdater(),
                is_continue :bool = False,
                start_index:int = 0):

    if new_model and initial_selfplay_repeat > 0 and initial_train_count > 0:
        initial_train(
            game_board=game_board,
            best_model_file= best_model_file,
            history_folder= history_folder,
            initial_selfplay_repeat= initial_selfplay_repeat,
            initial_train_count= initial_train_count,
            history_updater=history_updater)
    if eval_count > 0:
        executor = concurrent.futures.ThreadPoolExecutor(2)
    else:
        executor = None

    if use_cache:
        ts_dict = ThreadSafeDict()
    else:
        ts_dict = None

    for i in range(cycle_count):
        print('cycle {}/{}'.format(i + 1, cycle_count))
        if use_cache:
            ts_dict.clear()
        best_model = tf.keras.models.load_model(best_model_file)
        #first_brain = SelfplayNetworkBrain(evaluate_count=brain_evaluate_count, model=best_model, ts_dict=ts_dict)
        #second_brain = SelfplayNetworkBrain(evaluate_count=brain_evaluate_count, model=best_model, ts_dict=ts_dict)
        first_brain = NetworkBrainFactory.create_selfplay_dualmodel_network_brain(evaluate_count=brain_evaluate_count, first_model=best_model, second_model=best_model, ts_dict=ts_dict, history_updater=history_updater)
        second_brain = NetworkBrainFactory.create_selfplay_dualmodel_network_brain(evaluate_count=brain_evaluate_count, first_model=best_model, second_model=best_model, ts_dict=ts_dict, history_updater=history_updater)

        history_file = self_play(
            first_brain=first_brain,
            second_brain= second_brain,
            game_board= game_board,
            repeat_count= selfplay_repeat,
            history_folder= history_folder,
            is_continue=is_continue,
            start_index=start_index)
        is_continue = False
        start_index = 0
        latest_model, latest_file_name = train_network(best_model_file, history_file, game_board, epoch_count)

        print('training latest model file={0}'.format(latest_file_name))
        replace = True
        if eval_count > 0:
            if use_cache:
                latest_dict = ThreadSafeDict()
            else:
                latest_dict = None
            
            latest_brain = NetworkBrainFactory.create_network_brain(evaluate_count=brain_evaluate_count, model=latest_model, ts_dict=latest_dict)
            best_brain = NetworkBrainFactory.create_network_brain(evaluate_count=brain_evaluate_count, model=best_model, ts_dict=ts_dict)
            stats = evaluate_model(agent_target=Agent(brain=latest_brain, name='latest'), agent_base=Agent(brain=best_brain, name='best'), board=game_board,play_count=eval_count, executor=executor)
            replace = eval_judge(stats)
        if replace:
            os.remove(best_model_file)
            shutil.copy(latest_file_name, best_model_file)
            print("latest model replace best model")
        del best_model
        del latest_model
        tf.keras.backend.clear_session()

def initial_train_dual_model(game_board: GameBoard,
                        first_best_model_file: str,
                        second_best_model_file: str,
                        history_first_folder: str,
                        history_second_folder: str,
                        initial_selfplay_repeat: int,
                        initial_train_count: int,
                        executor: concurrent.futures.ThreadPoolExecutor,
                        history_updater: HistoryUpdater):
    print('initial selfplay and train')

    first,second = self_play_dualmodel(
        first_brain=SelfplayRandomBrain(history_updater=history_updater),
        second_brain= SelfplayRandomBrain(history_updater=history_updater),
        board=game_board,
        repeat_count=initial_selfplay_repeat,
        history_folder_first=history_first_folder,
        history_folder_second=history_second_folder)    
    print('initial selfplay completed. begin train')
    future_first = executor.submit(lambda: train_network(first_best_model_file, first, game_board, initial_train_count))
    future_second = executor.submit(lambda: train_network(second_best_model_file, second, game_board, initial_train_count))
    latest_first_model, latest_file_name_first = future_first.result()
    latest_second_model, latest_file_name_second = future_second.result()

    print('initial training complete. first model file={0}, second model file={1}'.format(latest_file_name_first, latest_file_name_second))
    os.remove(first_best_model_file)
    shutil.copy(latest_file_name_first, first_best_model_file)
    os.remove(second_best_model_file)
    shutil.copy(latest_file_name_second, second_best_model_file)
    del latest_first_model
    del latest_second_model
    tf.keras.backend.clear_session()

def train_cycle_dualmodel(
                game_board : GameBoard,
                brain_evaluate_count : int,
                first_best_model_file : str,
                second_best_model_file : str,
                history_first_folder : str,
                history_second_folder : str,
                selfplay_repeat : int = 500,
                epoch_count : int = 200,
                cycle_count : int = 10,
                eval_count: int = 20,
                eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats,
                use_cache = True,
                new_model: bool = False,
                initial_selfplay_repeat: int = 1000,
                initial_train_count: int = 500,
                history_updater: HistoryUpdater=HistoryUpdater(),
                is_continue :bool = False,
                start_index:int = 0):
    print('train_cycle_dualmodel')
    executor = concurrent.futures.ThreadPoolExecutor(2)

    if new_model and initial_selfplay_repeat > 0 and initial_train_count > 0:
        initial_train_dual_model(
            game_board=game_board,
            first_best_model_file= first_best_model_file,
            second_best_model_file= second_best_model_file,
            history_first_folder= history_first_folder,
            history_second_folder= history_second_folder,
            initial_selfplay_repeat= initial_selfplay_repeat,
            initial_train_count= initial_train_count,
            executor= executor,
            history_updater=history_updater)

    if use_cache:
        ts_dict = ThreadSafeDict()
    else:
        ts_dict = None
    for i in range(cycle_count):
        print('cycle {}/{}'.format(i + 1, cycle_count))
        if use_cache:
            ts_dict.clear()
        first_model = tf.keras.models.load_model(first_best_model_file)
        second_model = tf.keras.models.load_model(second_best_model_file)
        first_brain = NetworkBrainFactory.create_selfplay_dualmodel_network_brain(evaluate_count=brain_evaluate_count, first_model=first_model, second_model=second_model, ts_dict=ts_dict, history_updater=history_updater)
        second_brain = NetworkBrainFactory.create_selfplay_dualmodel_network_brain(evaluate_count=brain_evaluate_count, first_model=first_model, second_model=second_model, ts_dict=ts_dict, history_updater=history_updater)
        first_history_file, second_history_fine = self_play_dualmodel(first_brain, second_brain,game_board, selfplay_repeat, history_first_folder, history_second_folder)
        
        future_first = executor.submit(lambda: train_network(first_best_model_file, first_history_file, game_board, epoch_count))
        future_second = executor.submit(lambda: train_network(second_best_model_file, second_history_fine, game_board, epoch_count))
        latest_first_model, latest_file_name_first = future_first.result()
        latest_second_model, latest_file_name_second = future_second.result()

        print('training first model file={0}, second model file={1}'.format(latest_file_name_first, latest_file_name_second))

        replace = True
        if eval_count > 0:
            if use_cache:
                latest_dict = ThreadSafeDict()
            else:
                latest_dict = None            
            latest_brain = NetworkBrainFactory.create_dualmodel_network_brain(evaluate_count=brain_evaluate_count, first_model=latest_first_model, second_model=latest_second_model, ts_dict=latest_dict) 
            best_brain = NetworkBrainFactory.create_dualmodel_network_brain(evaluate_count=brain_evaluate_count, first_model=first_model, second_model=second_model, ts_dict=ts_dict) 
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
