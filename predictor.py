from game_board import GameBoard
from tensorflow.keras.models import Model
from threadsafe_dict import ThreadSafeDict
import numpy as np


class Prediction:
    
    #abstract method
    def get_policies(self)->np.ndarray:
        pass
    #abstract method
    def get_value(self)->float:
        pass


class DualNetworkPrediction:
    def __init__(self, policies: np.ndarray, value: float):
        self.policies = policies
        self.value = value

class PolicyNetworkPrediction:
    def __init__(self, policies: np.ndarray):
        self.policies = policies

class Predictor:
    def __init__(self, model: Model, ts_dict : ThreadSafeDict = None):
        self.model = model
        self.ts_dict = ts_dict
    def __call__(self, game_board: GameBoard) -> Prediction:
        return self.predict(game_board)
    
    def predict(self, game_board: GameBoard) -> Prediction:
        if self.ts_dict is None:
            return self.predict_core(game_board =game_board)
        return self.ts_dict.get_or_add(game_board.to_state_key(), lambda: self.predict_core(game_board=game_board))
    #abstract method
    def predict_core(self):
        pass

class DualNetworkPredictor(Predictor):
    def __init__(self, model: Model, ts_dict : ThreadSafeDict = None):
        super().__init__(model, ts_dict)


    def predict_core(self, game_board: GameBoard) -> Prediction:
        # 推論のための入力データのシェイプの変換
        x = game_board.reshape_to_input()
        # 推論
        y = self.model.predict(x, batch_size=1, verbose=0)
        # 方策の取得
        policies = y[0][0][:]
        policies /= np.sum(policies) if np.sum(policies) else 1 # 合計1の確率分布に変換
        # 価値の取得
        value = y[1][0][0]
        return DualNetworkPrediction(policies, value)

class PolicyNetworkPredictor(Predictor):
    def __init__(self, model: Model, ts_dict : ThreadSafeDict = None):
        super().__init__(model, ts_dict)

    def predict_core(self, game_board: GameBoard) -> Prediction:
        # 推論のための入力データのシェイプの変換
        x = game_board.reshape_to_input()
        # 推論
        y = self.model.predict(x, batch_size=1, verbose=0)
        return PolicyNetworkPrediction(y)


'''
def predict_core(model : Model, game_board : GameBoard)->Prediction:
    # 推論のための入力データのシェイプの変換
    x = game_board.reshape_to_input()
    # 推論
    y = model.predict(x, batch_size=1, verbose=0)
    # 方策の取得
    policies = y[0][0][:]
    policies /= np.sum(policies) if np.sum(policies) else 1 # 合計1の確率分布に変換
    # 価値の取得
    value = y[1][0][0]
    return policies, value

# 推論
def predict(model : Model, board : GameBoard, ts_dict : ThreadSafeDict)->Tuple[np.ndarray, float]:
    if ts_dict is None:
        return predict_core(model, board)
    return ts_dict.get_or_add(board.to_state_key(), lambda: predict_core(model, board))
'''
