from typing import Callable, Tuple
import numpy as np
from tensorflow.keras.models import Model
from brains import Brain
from game_board import GameBoard
from pv_mcts import pv_mcts_policies,pv_mcts_policies_boltzman
from selfplay_brain import SelfplayBrain
from threadsafe_dict import ThreadSafeDict

def predict_core(model : Model, board : GameBoard)->Tuple[np.ndarray, float]:
    # 推論のための入力データのシェイプの変換
    x = board.reshape_to_input()
    #print(x)
    # 推論
    y = model.predict(x, batch_size=1, verbose=0)

    with open('DEBUG_OUT.txt', 'a' ) as f:
        f.write("======= PREDICT ========\n")
        f.write('board={}\n'.format(board))
        f.write('p,v={}\n'.format(y))
        f.write('ptype={}\n'.format(y[0].dtype))
        f.write('vtype={}\n'.format(y[1].dtype))

    # 方策の取得
        policies = y[0][0][:]
        policies /= np.sum(policies) if np.sum(policies) else 1 # 合計1の確率分布に変換
        f.write('policies={}\n'.format(policies.dtype))


        # 価値の取得
        value = y[1][0][0]
    return policies, value

# 推論
def predict(model : Model, board : GameBoard, ts_dict : ThreadSafeDict)->Tuple[np.ndarray, float]:

    if ts_dict is None:
        return predict_core(model, board)
    return ts_dict.get_or_add(board.to_state_key(), lambda: predict_core(model, board))


class NetworkBrain(Brain):
    def __init__(self, evaluate_count : int, model: Model, ts_dict : ThreadSafeDict):
        super().__init__()
        self.evaluate_count = evaluate_count
        self.model = model
        self.last_policies = None
        self.ts_dict = ts_dict
    def get_name(self):
        return "NetworkBrain"
    def select_action(self, board)->int:
        ratios = pv_mcts_policies(board, self.evaluate_count, lambda x: predict(self.model, x, self.ts_dict) , lambda x: predict(self.model, x, self.ts_dict))
        self.last_policies = ratios
        action = np.argmax(ratios)
        return action
    def get_last_policies(self):
        return self.last_policies

class DualModelNetworkBrain(Brain):
    def __init__(self, evaluate_count : int, first_model : Model, second_model : Model, ts_dict : ThreadSafeDict):
        super().__init__()
        self.evaluate_count = evaluate_count
        self.first_model = first_model
        self.second_model = second_model
        self.last_policies = None
        self.ts_dict = ts_dict
    def get_name(self):
        return "DualModelNetworkBrain"
    
    def select_action(self, board)->int:
        if board.is_first_player_turn():
            ratios = pv_mcts_policies(board, self.evaluate_count, lambda x: predict(self.first_model, x, self.ts_dict) , lambda x: predict(self.second_model, x, self.ts_dict))
        else:
            ratios = pv_mcts_policies(board, self.evaluate_count, lambda x: predict(self.second_model, x, self.ts_dict) , lambda x: predict(self.first_model, x, self.ts_dict))
        self.last_policies = ratios
        action = np.argmax(ratios)
        return action
    def get_last_policies(self):
        return self.last_policies

class SelfplayNetworkBrain(SelfplayBrain):
    def __init__(self, evaluate_count : int, model: Model, ts_dict : ThreadSafeDict):
        super().__init__()
        self.evaluate_count = evaluate_count
        self.model = model
        self.ts_dict = ts_dict
    def get_name(self):
        return "SelfplayNetworkBrain"
    def select_action(self, board : GameBoard)->int:
        policies = pv_mcts_policies_boltzman(board, self.evaluate_count, lambda x: predict(self.model, x, self.ts_dict) , lambda x: predict(self.model, x, self.ts_dict))
        action = np.random.choice(range(0, board.get_output_size()), p=policies)
        self.register_policies(board, policies)
        return action
    
class SelfplayDualModelNetworkBrain(SelfplayBrain):
    def __init__(self, evaluate_count : int, first_model : Model, second_model : Model, ts_dict : ThreadSafeDict):
        super().__init__()
        self.evaluate_count = evaluate_count
        self.first_model = first_model
        self.second_model = second_model
        self.last_policies = None
        self.ts_dict = ts_dict
    def get_name(self):
        return "SelfplayDualModelNetworkBrain"
    def select_action(self, board : GameBoard)->int:
        if board.is_first_player_turn():
            policies = pv_mcts_policies_boltzman(board, self.evaluate_count, lambda x: predict(self.first_model, x, self.ts_dict) , lambda x: predict(self.second_model, x, self.ts_dict))
        else:
            policies = pv_mcts_policies_boltzman(board, self.evaluate_count, lambda x: predict(self.second_model, x, self.ts_dict) , lambda x: predict(self.first_model, x, self.ts_dict))
        action = np.random.choice(range(0, board.get_output_size()), p=policies)
        self.register_policies(board, policies)
        return action
