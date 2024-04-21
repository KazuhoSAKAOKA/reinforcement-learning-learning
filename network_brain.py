from typing import Callable, Tuple
import numpy as np
from tensorflow.keras.models import Model

from game_board import GameBoard
from pv_mcts import pv_mcts_scores
from selfplay_brain import SelfplayBrain


# 推論
def predict(model : Model, board : GameBoard)->Tuple[np.ndarray, float]:
    # 推論のための入力データのシェイプの変換
    x = board.get_model_input_shape()
    #print(x)
    # 推論
    y = model.predict(x, batch_size=1, verbose=0)

    # 方策の取得
    policies = y[0][0][list(board.get_legal_actions())] # 合法手のみ
    policies /= sum(policies) if sum(policies) else 1 # 合計1の確率分布に変換

    # 価値の取得
    value = y[1][0][0]
    return policies, value



class NetworkBrain:
    def __init__(self, temperature: float, evaluate_count : int, model: Model):
        self.temperature = temperature
        self.evaluate_count = evaluate_count
        self.model = model
        self.last_policies = None
    def get_name(self):
        return "NetworkBrain"
    
    def select_action(self, board)->int:
        scores = pv_mcts_scores(board, self.temperature, self.evaluate_count, lambda x: predict(self.model, x) , lambda x: predict(self.model, x))
        self.last_policies = scores
        action = np.random.choice(board.get_legal_actions(), p=scores)
        return action
    def get_last_policies(self):
        return self.last_policies

class DualModelNetworkBrain:
    def __init__(self, temperature: float, evaluate_count : int, first_model : Model, second_model : Model):
        self.temperature = temperature
        self.evaluate_count = evaluate_count
        self.first_model = first_model
        self.second_model = second_model
        self.last_policies = None
    def get_name(self):
        return "DualModelNetworkBrain"
    
    def select_action(self, board)->int:
        if board.is_first_player_turn():
            scores = pv_mcts_scores(board, self.temperature, self.evaluate_count, lambda x: predict(self.first_model, x) , lambda x: predict(self.second_model, x))
        else:
            scores = pv_mcts_scores(board, self.temperature, self.evaluate_count, lambda x: predict(self.second_model, x) , lambda x: predict(self.first_model, x))
        self.last_policies = scores
        action = np.random.choice(board.get_legal_actions(), p=scores)
        return action
    def get_last_policies(self):
        return self.last_policies
    

class SelfplayNetworkBrain(SelfplayBrain):
    def __init__(self, temperature: float, evaluate_count : int, model: Model):
        super().__init__()
        self.temperature = temperature
        self.evaluate_count = evaluate_count
        self.model = model
    def get_name(self):
        return "SelfplayNetworkBrain"
    def select_action(self, board : GameBoard)->int:
        scores = pv_mcts_scores(board, self.temperature, self.evaluate_count, lambda x: predict(self.model, x) , lambda x: predict(self.model, x))
        action = np.random.choice(board.get_legal_actions(), p=scores)
        self.register_action(board, action)
        return action
    
class SelfplayDualModelNetworkBrain(SelfplayBrain):
    def __init__(self, temperature: float, evaluate_count : int, first_model : Model, second_model : Model):
        super().__init__()
        self.temperature = temperature
        self.evaluate_count = evaluate_count
        self.first_model = first_model
        self.second_model = second_model
        self.last_policies = None
    def get_name(self):
        return "SelfplayDualModelNetworkBrain"
    def select_action(self, board : GameBoard)->int:
        if board.is_first_player_turn():
            scores = pv_mcts_scores(board, self.temperature, self.evaluate_count, lambda x: predict(self.first_model, x) , lambda x: predict(self.second_model, x))
        else:
            scores = pv_mcts_scores(board, self.temperature, self.evaluate_count, lambda x: predict(self.second_model, x) , lambda x: predict(self.first_model, x))
        action = np.random.choice(board.get_legal_actions(), p=scores)
        self.register_action(board, action)
        return action