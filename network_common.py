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
from self_play import self_play_impl
from predictor import DualNetworkPredictor 
from self_play import HistoryData

'''
model file フォルダ構成
./model/
    game_name/
        yyyyMMddHHmmss/     # 処理開始
            gen0000.keras   # 0世代目
        best.keras          # 最新のモデル
        best_first.keras    # 先行と後攻のモデルが別の場合の最新のモデル
        best_second.keras   # 先行と後攻のモデルが別の場合の最新のモデル 
'''

BEST_MODEL_FILE = 'best.keras'

BEST_FIRST_PLAYER_MODEL_FILE = 'best_first.keras'
BEST_SECOND_PLAYER_MODEL_FILE = 'best_second.keras'

def try_crate_model_file(model_file: str, create_model: Callable[[],Model])->bool:
    if os.path.exists(model_file):
        return False
    model = create_model()
    model.save(model_file)
    del model
    tf.keras.backend.clear_session()
    return True

def train_network(
        load_model_path : str, 
        train_model_folder : str,
        generation : int,
        model_file_postfix : str,
        history_data : HistoryData, 
        game_board : GameBoard, 
        epoch_count : int)->Tuple[Model, str]:

    model = tf.keras.models.load_model(load_model_path)
    history = history_data.deserialize()
    history = game_board.augmente_data(history)
    xs, y_policies, y_values = game_board.reshape_history_to_input(history)

#    for p in y_policies:
#        for q in p:
#            if np.isnan(q):
#                print('train data nan!' )
    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    print_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: print('\rTrain {}/{}'.format(epoch + 1, epoch_count), end=''))

    model.fit(xs, [y_policies, y_values], batch_size=128, epochs=epoch_count, 
              verbose=0, callbacks=[lr_decay, print_callback])
    if model_file_postfix is None:
        save_path = train_model_folder + '/gen{:04}.keras'.format(generation)
    else:
        save_path = train_model_folder + '/gen{:04}_{}.keras'.format(generation, model_file_postfix)
    print('\rcomplete. train file={}'.format(save_path))
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

def create_best_model_file_path(network_param: NetworkParameter)->Tuple[str,str]:
    os.makedirs(network_param.model_folder, exist_ok=True)
    best_model_file:str
    best_model_file_second:str
    if network_param.is_dual_model:
        best_model_file = network_param.model_folder +'/' + BEST_FIRST_PLAYER_MODEL_FILE
        best_model_file_second = network_param.model_folder +'/' + BEST_SECOND_PLAYER_MODEL_FILE
    else:
        best_model_file = network_param.model_folder +'/' + BEST_MODEL_FILE
        best_model_file_second = None
    return best_model_file, best_model_file_second

def initial_train(
            game_board: GameBoard,
            network_param: NetworkParameter,
            selfplay_param: SelfplayParameter,
            brain_param: BrainParameter,
            initial_selfplay_param:InitSelfplayParameter,
            executor: concurrent.futures.ThreadPoolExecutor,
            train_model_folder : str):
    print('initial selfplay and train')
    temp_param = copy.copy(selfplay_param)
    temp_param.selfplay_repeat = initial_selfplay_param.selfplay_repeat
    temp_param.train_epoch = initial_selfplay_param.train_epoch
    history_data = self_play_impl(
        first_brain=SelfplayRandomMCTSBrain(brain_param=brain_param),
        second_brain= SelfplayRandomMCTSBrain(brain_param=brain_param),
        game_board=game_board,
        is_dual_model=network_param.is_dual_model,
        selfplay_param=temp_param)
    print('initial selfplay completed. begin train')
    best_model_file, best_model_file_second = create_best_model_file_path(network_param)

    if network_param.is_dual_model:
        future_first = executor.submit(lambda: train_network(
            load_model_path=best_model_file,
            train_model_folder=train_model_folder,
            generation=0,
            model_file_postfix='first',
            history_data=history_data.get_primary(),
            game_board=game_board,
            epoch_count=initial_selfplay_param.train_epoch))
        future_second = executor.submit(lambda: train_network(
            load_model_path=best_model_file_second,
            train_model_folder=train_model_folder,
            generation=0,
            model_file_postfix='second',
            history_data=history_data.get_secondary(),
            game_board=game_board,
            epoch_count=initial_selfplay_param.train_epoch))
    else:
        future_first = executor.submit(lambda: train_network(
            load_model_path=best_model_file,
            train_model_folder=train_model_folder,
            generation=0,
            model_file_postfix=None,
            history_data=history_data.get_primary(),
            game_board=game_board,
            epoch_count=initial_selfplay_param.train_epoch))
        future_second = None

    latest_first_model, latest_file_name_first = future_first.result()
    if network_param.is_dual_model:
        latest_second_model, latest_file_name_second = future_second.result()    
 
    print('initial training complete. model file={}'.format(latest_file_name_first))
    if network_param.is_dual_model:
        print('initial training complete. model file={}'.format(latest_file_name_second))

    os.remove(best_model_file)
    shutil.copy(latest_file_name_first, best_model_file)

    if network_param.is_dual_model:
        os.remove(best_model_file_second)
        shutil.copy(latest_file_name_second, best_model_file_second)

    del latest_first_model
    if network_param.is_dual_model:
        del latest_second_model
    tf.keras.backend.clear_session()

def train_cycle(
                game_board : GameBoard,
                network_param: NetworkParameter,
                create_model: Callable[[], Model],
                selfplay_param: SelfplayParameter,
                brain_param: BrainParameter,
                exploration_param: ExplorationParameter,
                initial_selfplay_param: InitSelfplayParameter = None):

    best_model_file, best_model_file_second = create_best_model_file_path(network_param)
    if network_param.is_dual_model:
        created_first = try_crate_model_file(best_model_file, create_model)
        if created_first:
            print('create best model file:{}'.format(best_model_file))
        created_second = try_crate_model_file(best_model_file_second, create_model)
        if created_second:
            print('create best model file:{}'.format(best_model_file_second))
        created = created_first or created_second
    else:
        created = try_crate_model_file(best_model_file, create_model)
        if created:
            print('create best model file:{}'.format(best_model_file))

    executor = concurrent.futures.ThreadPoolExecutor(2)

    now = datetime.now()
    train_model_folder = network_param.model_folder + '/{:04}{:02}{:02}{:02}{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    os.makedirs(train_model_folder, exist_ok=True)
    if created:
        if initial_selfplay_param is not None:
            initial_train(
                game_board=game_board,
                network_param=network_param,
                selfplay_param=selfplay_param,
                brain_param=brain_param,
                initial_selfplay_param=initial_selfplay_param,
                executor=executor,
                train_model_folder=train_model_folder)

    if brain_param.use_cache:
        ts_dict = ThreadSafeDict()
    else:
        ts_dict = None

    for i in range(selfplay_param.cycle_count):
        print('cycle {}/{}'.format(i + 1, selfplay_param.cycle_count))
        if brain_param.use_cache:
            ts_dict.clear()

        # selfplay brainはhistoryデータを各ブレインで持っているので先手後手で異なるインスタンスを使う
        first_model = tf.keras.models.load_model(best_model_file)

        if network_param.is_dual_model:
            second_model = tf.keras.models.load_model(best_model_file_second)
            predictor_first = DualNetworkPredictor(model=first_model, ts_dict=ts_dict)
            predictor_second = DualNetworkPredictor(model=second_model, ts_dict=ts_dict)
            first_brain = NetworkBrainFactory.create_selfplay_dualmodel_network_brain(
                    predictor_first= predictor_first,
                    predictor_second= predictor_second,
                    brain_param=brain_param,
                    exploration_param=exploration_param)
            second_brain = NetworkBrainFactory.create_selfplay_dualmodel_network_brain(
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
            
        history_data = self_play_impl(
            first_brain=first_brain, 
            second_brain=second_brain, 
            game_board=game_board, 
            is_dual_model=network_param.is_dual_model,
            selfplay_param=selfplay_param)

        if network_param.is_dual_model:
            future_first = executor.submit(lambda: train_network(
                load_model_path=best_model_file,
                train_model_folder=train_model_folder,
                generation=i+1,
                model_file_postfix='first',
                history_data=history_data.get_primary(),
                game_board=game_board,
                epoch_count=initial_selfplay_param.train_epoch))
            future_second = executor.submit(lambda: train_network(
                load_model_path=best_model_file_second,
                train_model_folder=train_model_folder,
                generation=i+1,
                model_file_postfix='second',
                history_data=history_data.get_secondary(),
                game_board=game_board,
                epoch_count=initial_selfplay_param.train_epoch))
        else:
            future_first = executor.submit(lambda: train_network(
                load_model_path=best_model_file,
                train_model_folder=train_model_folder,
                generation=i+1,
                model_file_postfix=None,
                history_data=history_data.get_primary(),
                game_board=game_board,
                epoch_count=initial_selfplay_param.train_epoch))
            future_second = None
                    
        latest_first_model, latest_file_name_first = future_first.result()
        if network_param.is_dual_model:
            latest_second_model, latest_file_name_second = future_second.result()

        if network_param.is_dual_model:
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

            if network_param.is_dual_model:
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
            os.remove(best_model_file)
            shutil.copy(latest_file_name_first, best_model_file)
            print("first model replace best model")
            if network_param.is_dual_model:
                os.remove(best_model_file_second)
                shutil.copy(latest_file_name_second, best_model_file_second)
                print("second model replace best model") 
        del latest_first_model
        del first_model
        if network_param.is_dual_model:
            del latest_second_model
            del second_model
        tf.keras.backend.clear_session()

