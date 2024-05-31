# パラメータ初期


#C_PUCT = 2.5
from enum import Enum


C_BASE = 19652
C_INIT = 1.25

# ディリクレノイズ
ALPHA = 0.3
EPSILON = 0.25

# 割引率
GAMMA = 0.99
# ボルツマン定数温度
TEMPERATURE = 1.0

class NetworkType(Enum):
    DualNetwork = 0,
    PolicyNetwork = 1,
    ValueNetwork = 2,

class HistoryUpdateType(Enum):
    constants = 0
    discount_gamma = 1
    zero_to_one = 2


class Parameter:
    def __init__(self, 
                c_base : float=C_BASE, 
                c_init : float=C_INIT,
                alpha : float=ALPHA,
                epsilon : float=EPSILON,
                history_update_type:HistoryUpdateType=HistoryUpdateType.zero_to_one,
                gamma : float = GAMMA,
                temperature : float = TEMPERATURE,
                network_type : NetworkType = NetworkType.DualNetwork,
                mcts_evaluate_count : int = 1000,
                mcts_expand_limit : int = 10):
        self.c_base = c_base
        self.c_init = c_init
        self.alpha = alpha
        self.epsilon = epsilon
        self.history_update_type = history_update_type
        self.gamma = gamma
        self.temperature = temperature
        self.network_type = network_type
        self.mcts_evaluate_count = mcts_evaluate_count
        self.mcts_expand_limit = mcts_expand_limit
PARAM = Parameter(
                c_base=C_BASE,
                c_init= C_INIT,
                alpha= ALPHA,
                epsilon= EPSILON,
                history_update_type=HistoryUpdateType.zero_to_one,
                gamma= GAMMA,
                temperature= TEMPERATURE,
                network_type= NetworkType.DualNetwork,
                mcts_evaluate_count=50)

