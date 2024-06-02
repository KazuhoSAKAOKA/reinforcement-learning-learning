import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from datetime import datetime
import numpy as np
import os
import shutil
import concurrent.futures
import copy

from game_board import GameBoard, get_first_player_value
from game import GameEnv, GameStats
from agent import Agent
from brains import RandomBrain
from montecarlo import MonteCarloBrain
from typing import Callable
from parameter import ExplorationParameter, BrainParameter, SelfplayParameter,judge_stats, NetworkParameter,InitSelfplayParameter
from typing import Tuple
from self_play_brain import SelfplayRandomMCTSBrain
from network_brain import NetworkBrain, SelfplayNetworkBrain, NetworkBrainFactory
from threadsafe_dict import ThreadSafeDict
from self_play import self_play_impl, load_data_file_name
from predictor import DualNetworkPredictor 



def train_network(
        load_model_path : str, 
        history_file : str, 
        game_board : GameBoard, 
        epoch_count : int)->Tuple[Model, str]:

    model = tf.keras.models.load_model(load_model_path)

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

def evaluate_model(
        agent_target : Agent,
        agent_base: Agent, 
        game_board : GameBoard, 
        play_count: int, 
        executor:concurrent.futures.ThreadPoolExecutor = None)->Tuple[GameStats, GameStats]:
    print('begin evaluate best vs latest')

    if executor is None:
        env = GameEnv(game_board, agent_target, agent_base, episode_callback=lambda i, _1, _2: print('\rEvaluate {}/{}'.format(i + 1, play_count), end=''))
        first = env.play_n(play_count)

        env = GameEnv(game_board, agent_base, agent_target, episode_callback=lambda i, _1, _2: print('\rEvaluate {}/{}'.format(i + 1, play_count), end=''))
        second = env.play_n(play_count)

        return first, second
    else:
        def execute(agent_target, agent_base, game_board, play_count):
            env = GameEnv(game_board, agent_target, agent_base)
            return env.play_n(play_count)

        future_first = executor.submit(lambda: execute(agent_target, agent_base, game_board, play_count))
        future_second = executor.submit(lambda: execute(agent_base, agent_target, game_board, play_count))
        return future_first.result(), future_second.result()

def initial_train(
            game_board: GameBoard,
            network_param: NetworkParameter,
            selfplay_param: SelfplayParameter,
            brain_param: BrainParameter,
            initial_selfplay_param:InitSelfplayParameter,
            executor: concurrent.futures.ThreadPoolExecutor):
    print('initial selfplay and train')
    temp_param = copy.copy(selfplay_param)
    temp_param.selfplay_repeat = initial_selfplay_param.selfplay_repeat
    temp_param.train_epoch = initial_selfplay_param.train_epoch
    history_files = self_play_impl(
        first_brain=SelfplayRandomMCTSBrain(brain_param=brain_param),
        second_brain= SelfplayRandomMCTSBrain(brain_param=brain_param),
        game_board=game_board,
        selfplay_param=temp_param)
    print('initial selfplay completed. begin train')

    future_first = executor.submit(lambda: train_network(network_param.best_model_file, history_files[0], game_board, initial_selfplay_param.train_epoch))
    is_dual = network_param.best_model_file_second is not None
    if is_dual:
        future_second = executor.submit(lambda: train_network(network_param.best_model_file_second, history_files[1], game_board, selfplay_param.train_epoch))
    else:
        future_second = None
    latest_first_model, latest_file_name_first = future_first.result()
    if is_dual:
        latest_second_model, latest_file_name_second = future_second.result()    
 
    print('initial training complete. model file={}'.format(latest_file_name_first))
    if is_dual:
        print('initial training complete. model file={}'.format(latest_file_name_second))

    os.remove(network_param.best_model_file)
    shutil.copy(latest_file_name_first, network_param.best_model_file)

    if is_dual:
        os.remove(network_param.best_model_file_second)
        shutil.copy(latest_file_name_second, network_param.best_model_file_second)

    del latest_first_model
    if is_dual:
        del latest_second_model
    tf.keras.backend.clear_session()

'''
def initial_train_dual_model(game_board: GameBoard,
                        first_best_model_file: str,
                        second_best_model_file: str,
                        history_first_folder: str,
                        history_second_folder: str,
                        initial_selfplay_repeat: int,
                        initial_train_count: int,
                        executor: concurrent.futures.ThreadPoolExecutor,
                        param: Parameter):
    print('initial selfplay and train')

    first,second = self_play_dualmodel(
        first_brain=SelfplayRandomMCTSBrain(param=param),
        second_brain= SelfplayRandomMCTSBrain(param=param),
        game_board=game_board,
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
'''

def train_cycle(
                game_board : GameBoard,
                network_param: NetworkParameter,
                create_model_file: Callable[[str], bool],
                selfplay_param: SelfplayParameter,
                brain_param: BrainParameter,
                exploration_param: ExplorationParameter,
                initial_selfplay_param: InitSelfplayParameter = None):
    
    is_dual = network_param.best_model_file_second is not None

    created_first = create_model_file(network_param.best_model_file)
    if network_param.best_model_file_second is not None:
        created_second = create_model_file(network_param.best_model_file_second)
    else:
        created_second = False

    executor = concurrent.futures.ThreadPoolExecutor(2)

    if created_first:
        print('create best model file:{}'.format(network_param.best_model_file))
        if initial_selfplay_param is not None:
            initial_train(
                game_board=game_board,
                network_param=network_param,
                selfplay_param=selfplay_param,
                brain_param=brain_param,
                initial_selfplay_param=initial_selfplay_param,
                executor=executor)

    if brain_param.use_cache:
        ts_dict = ThreadSafeDict()
    else:
        ts_dict = None

    for i in range(selfplay_param.cycle_count):
        print('cycle {}/{}'.format(i + 1, selfplay_param.cycle_count))
        if brain_param.use_cache:
            ts_dict.clear()

        # selfplay brainはhistoryデータを各ブレインで持っているので先手後手で異なるインスタンスを使う
        first_model = tf.keras.models.load_model(network_param.best_model_file)
        if is_dual:
            second_model = tf.keras.models.load_model(network_param.best_model_file_second)
            predictor_first = DualNetworkPredictor(model=first_model, ts_dict=ts_dict)
            predictor_second = DualNetworkPredictor(model=second_model, ts_dict=ts_dict)
            first_brain = NetworkBrainFactory.create_selfplay_network_brain(
                    predictor_first= predictor_first,
                    predictor_second= predictor_second,
                    brain_param=brain_param,
                    exploration_param=exploration_param)
            second_brain = NetworkBrainFactory.create_selfplay_network_brain(
                    predictor_first= predictor_first,
                    predictor_second= predictor_second,
                    brain_param=brain_param,
                    exploration_param=exploration_param)
        else:
            predictor = DualNetworkPredictor(model=first_model, ts_dict=ts_dict)
            second_model = None
            first_brain = NetworkBrainFactory.create_selfplay_network_brain(
                    predictor= predictor,
                    brain_param=brain_param,
                    exploration_param=exploration_param)
            second_brain = NetworkBrainFactory.create_selfplay_network_brain(
                    predictor= predictor,
                    brain_param=brain_param,
                    exploration_param=exploration_param)
            
        history_files = self_play_impl(first_brain=first_brain, second_brain=second_brain, game_board=game_board, selfplay_param=selfplay_param)

        future_first = executor.submit(lambda: train_network(network_param.best_model_file, history_files[0], game_board, selfplay_param.train_epoch))
        if is_dual:
            future_second = executor.submit(lambda: train_network(network_param.best_model_file_second, history_files[1], game_board, selfplay_param.train_epoch))

        latest_first_model, latest_file_name_first = future_first.result()
        if is_dual:
            latest_second_model, latest_file_name_second = future_second.result()

        if is_dual:
            print('training first model file={0}, second model file={1}'.format(latest_file_name_first, latest_file_name_second))
        else:
            print('training model file={0}'.format(latest_file_name_first))

        replace = True
        if selfplay_param.evaluate_count > 0:
            # 新しく学習したモデルでは異なる辞書を使用
            if brain_param.use_cache:
                latest_dict = ThreadSafeDict()
            else:
                latest_dict = None

            if is_dual:
                best_predictor_first = DualNetworkPredictor(model=first_model, ts_dict=ts_dict)
                best_predictor_second = DualNetworkPredictor(model=second_model, ts_dict=ts_dict)
                best_brain = NetworkBrainFactory.create_dualmodel_network_brain(
                        predictor_first= best_predictor_first,
                        predictor_second= best_predictor_second,
                        brain_param=brain_param,
                        exploration_param=exploration_param)

                latest_predictor_first = DualNetworkPredictor(model=latest_first_model, ts_dict=ts_dict)
                latest_predictor_second = DualNetworkPredictor(model=latest_second_model, ts_dict=ts_dict)
                latest_brain = NetworkBrainFactory.create_dualmodel_network_brain(
                        predictor_first= latest_predictor_first,
                        predictor_second= latest_predictor_second,
                        brain_param=brain_param,
                        exploration_param=exploration_param)
            else:
                best_predictor = DualNetworkPredictor(model=first_model, ts_dict=ts_dict)
                latest_predictor = DualNetworkPredictor(model=latest_first_model, ts_dict=latest_dict)
                best_brain = NetworkBrainFactory.create_network_brain(
                        predictor = best_predictor,
                        brain_param=brain_param,
                        exploration_param=exploration_param)
                latest_brain = NetworkBrainFactory.create_dualmodel_network_brain(
                        predictor_first= latest_predictor,
                        predictor_second= latest_predictor,
                        brain_param=brain_param,
                        exploration_param=exploration_param)                

            stats = evaluate_model(agent_target=Agent(brain=latest_brain, name='latest'), agent_base=Agent(brain=best_brain, name='best'), game_board=game_board,play_count=selfplay_param.evaluate_count, executor=executor)
            replace = selfplay_param.eval_judge(stats)
        if replace:
            os.remove(network_param.best_model_file)
            shutil.copy(latest_file_name_first, network_param.best_model_file)
            print("first model replace best model")
            if is_dual:
                os.remove(network_param.best_model_file_second)
                shutil.copy(latest_file_name_second, network_param.best_model_file_second)
                print("second model replace best model") 
        del latest_first_model
        del first_model
        if is_dual:
            del latest_second_model
            del second_model
        tf.keras.backend.clear_session()

'''
def train_cycle_dualmodel(
                game_board : GameBoard,
                first_best_model_file : str,
                second_best_model_file : str,
                create_model_file: Callable[[str], bool],
                history_first_folder : str,
                history_second_folder : str,
                brain_evaluate_count : int,
                selfplay_repeat : int = 500,
                epoch_count : int = 200,
                cycle_count : int = 10,
                eval_count: int = 20,
                eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats,
                use_cache = True,
                initial_selfplay_repeat: int = 1000,
                initial_train_count: int = 500,
                param: Parameter = Parameter(),
                is_continue :bool = False,
                start_index:int = 0):
    print('train_cycle_dualmodel')
    executor = concurrent.futures.ThreadPoolExecutor(2)
    new_model_first = create_model_file(first_best_model_file)
    new_model_second = create_model_file(second_best_model_file)
    if new_model_first and new_model_second and initial_selfplay_repeat > 0 and initial_train_count > 0:
        initial_train_dual_model(
            game_board=game_board,
            first_best_model_file= first_best_model_file,
            second_best_model_file= second_best_model_file,
            history_first_folder= history_first_folder,
            history_second_folder= history_second_folder,
            initial_selfplay_repeat= initial_selfplay_repeat,
            initial_train_count= initial_train_count,
            executor= executor,
            param=param)

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
'''