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
from game import GameEnv
from agent import Agent
from brains import RandomBrain
from montecarlo import MonteCarloBrain

def write_data(folder, history):
    now = datetime.now()
    folder = '{0}/'.format(folder)
    os.makedirs(folder, exist_ok=True)
    path = folder + '/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)


# 推論
def predict(model, board):
    # 推論のための入力データのシェイプの変換
    x = board.get_model_input_shape()

    # 推論
    y = model.predict(x, batch_size=1, verbose=0)

    # 方策の取得
    policies = y[0][0][list(board.get_legal_actions())] # 合法手のみ
    policies /= sum(policies) if sum(policies) else 1 # 合計1の確率分布に変換

    # 価値の取得
    value = y[1][0][0]
    return policies, value

class SelfplayNetworkBrain:
    def __init__(self, temperature, evaluate_count, self_predict, enemy_predict):
        self.temperature = temperature
        self.evaluate_count = evaluate_count
        self.history = []
        self.self_predict = self_predict
        self.enemy_predict = enemy_predict
    def get_name(self):
        return "SelfplayNetworkBrain"
    
    def select_action(self, board):
        scores = pv_mcts_scores(board, self.temperature, self.evaluate_count, self.self_predict, self.enemy_predict)
        policies = [0] * board.get_output_size()
        for action, policy in zip(board.get_legal_actions(), scores):
            policies[action] = policy
        self.history.append([board.get_model_state(), policies, None])
        action = np.random.choice(board.get_legal_actions(), p=scores)
        return action
    def update_history(self, value, gammma):
        i = len(self.history) - 1
        while i >= 0:
            self.history[i][2] = value
            value = gammma * value
            i -= 1
    def reset(self):
        self.history = []
class NetworkBrain:
    def __init__(self, temperature, evaluate_count, self_predict, enemy_predict):
        self.temperature = temperature
        self.evaluate_count = evaluate_count
        self.self_predict = self_predict
        self.enemy_predict = enemy_predict
        self.last_policies = None
    def get_name(self):
        return "NetworkBrain"
    
    def select_action(self, board):
        scores = pv_mcts_scores(board, self.temperature, self.evaluate_count, self.self_predict, self.enemy_predict)
        self.last_policies = scores
        action = np.random.choice(board.get_legal_actions(), p=scores)
        return action
    def get_last_policies(self):
        return self.last_policies

def load_data(history_folder):
    history_path = sorted(Path(history_folder).glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)
    
def self_play(model_file, history_folder, board, temperature, repeat_count, gamma = 0.99):

    model = load_model(model_file)

    history=[]

    def do_empty1(current_board):
        pass
    def do_empty2(board, action):
        pass
    def update_history(_1, board, result):
        value = get_first_player_value(result)
        first_agent.brain.update_history(value, gamma)
        second_agent.brain.update_history(-value, gamma)
        history.extend(first_agent.brain.history)
        history.extend(second_agent.brain.history)
        #history.append([board.get_model_state(), [0] * board.get_output_size(), value])

    first_agent = Agent(SelfplayNetworkBrain(temperature, 10, lambda x: predict(model, x), lambda x: predict(model, x)))
    second_agent = Agent(SelfplayNetworkBrain(temperature, 10, lambda x: predict(model, x), lambda x: predict(model, x)))

    env = GameEnv(board, first_agent, second_agent, do_empty1, do_empty2, update_history)
    for i in range(repeat_count):
        result = env.play()
        first_agent.brain.reset()
        second_agent.brain.reset()
        print('\rSelf play {}/{}'.format(i + 1, repeat_count), end='')
    write_data(history_folder, history)
    K.clear_session()
    del model
    print('\rcomplete. self play')
    return history

    
def train_network(load_model_path, save_model_path, history_folder, game_board : GameBoard, epoch_count : int):

    model = load_model(load_model_path)
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    history = load_data(history_folder)
    xs, y_policies, y_values = game_board.convert_history_to_model_input(history)

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
    save_file_name = path = save_model_path + '/{:04}{:02}{:02}{:02}{:02}{:02}.keras'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    model.save(save_file_name)

    K.clear_session()
    del model
    return save_file_name

def evalute_best_model(best_model_file, latest_model_file, board, play_count):
    best_model = load_model(best_model_file)
    latest_model = load_model(latest_model_file)
    champion_agent = Agent(NetworkBrain(1.0, 50, lambda x: predict(best_model, x), lambda x: predict(latest_model, x)))
    challenger_agent = Agent(NetworkBrain(1.0, 50, lambda x: predict(latest_model, x), lambda x: predict(best_model, x)))

    env = GameEnv(board, challenger_agent, champion_agent, episode_callback=lambda i, _1, _2: print('\rEvaluate {}/{}'.format(i + 1, play_count), end=''))
    latest_win_first, latest_lose_first, latest_draw_first = env.play_n(play_count)
    latest_first_point = latest_win_first * 2 + latest_draw_first

    env = GameEnv(board, champion_agent, challenger_agent, episode_callback=lambda i, _1, _2: print('\rEvaluate {}/{}'.format(i + 1, play_count), end=''))
    latest_lose_second, latest_win_second, latest_draw_second = env.play_n(play_count)
    latest_second_point = latest_win_second * 2 + latest_draw_second

    best_first_point = latest_lose_second * 2 + latest_draw_second
    best_second_point = latest_lose_first * 2 + latest_draw_first

    K.clear_session()
    del best_model
    del latest_model
    print('point={0},{1},{2},{3}'.format(latest_first_point, best_first_point, latest_second_point, best_second_point))
    if latest_first_point >= best_first_point and latest_second_point > best_second_point:
        print("New champion")
        os.remove(best_model_file)
        os.rename(latest_model_file, best_model_file)
        return True
    print('\rcomplete. evaluate best model')
    return False


def train_cycle(best_model_file, save_model_path, history_folder, convert, self_play_repeat, epoch_count, cycle_count):
    for i in range(cycle_count):
        print('cycle {}/{}'.format(i + 1, cycle_count))
        self_play(best_model_file, history_folder, TicTacToeBoard(), 1.0, self_play_repeat)
        latest_file_name = train_network(best_model_file, save_model_path, history_folder, convert, epoch_count)
        evalute_best_model(best_model_file, latest_file_name, TicTacToeBoard(), 50)

if __name__ == '__main__':
    pass