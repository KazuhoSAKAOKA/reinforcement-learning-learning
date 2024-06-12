# パラメータ初期


#C_PUCT = 2.5
from enum import Enum
from typing import Callable, Tuple
from game import GameStats, judge_stats

C_BASE = 19652
C_INIT = 1.25

# ディリクレノイズ
ALPHA = 0.3
EPSILON = 0.25

# 割引率
GAMMA = 0.99
# ボルツマン定数温度
TEMPERATURE = 1.0


class HistoryUpdateType(Enum):
    constants = 0
    discount_gamma = 1
    zero_to_one = 2

class ActionSelectorType(Enum):
    max = 0
    random = 1
    boltzmann = 2

class BrainParameter:
    def __init__(self, 
                mcts_evaluate_count : int = 50,
                mcts_expand_limit : int = 10,
                use_cache = True,
                history_update_type:HistoryUpdateType=HistoryUpdateType.zero_to_one,
                action_selector_type:ActionSelectorType=ActionSelectorType.max,
                ):
        self.use_cache = use_cache
        self.mcts_evaluate_count = mcts_evaluate_count
        self.mcts_expand_limit = mcts_expand_limit
        self.history_update_type = history_update_type
        self.action_selector_type = action_selector_type
class NetworkType(Enum):
    DualNetwork = 0,
    PolicyNetwork = 1,
    ValueNetwork = 2,

class NetworkParameter:
    def __init__(self, 
                model_folder : str,
                network_type : NetworkType = NetworkType.DualNetwork,
                is_dual_model:bool = False):
        self.model_folder = model_folder
        self.network_type = network_type
        self.is_dual_model = is_dual_model

class SelfplayParameter:
    def __init__(self, 
                history_folder : str,
                cycle_count:int = 10,
                selfplay_repeat : int = 500,
                continue_history_folder_path :str = None,
                evaluate_count : int = 50,
                eval_judge: Callable[[Tuple[GameStats, GameStats]], bool] = judge_stats,
                gamma : float = GAMMA,
                train_epoch : int = 200,
                output_visualize_text : bool = True):
        self.cycle_count = cycle_count
        self.history_folder = history_folder
        self.selfplay_repeat = selfplay_repeat
        self.continue_history_folder_path = continue_history_folder_path
        self.evaluate_count = evaluate_count
        self.eval_judge = eval_judge
        self.gamma = gamma
        self.train_epoch = train_epoch
        self.output_visualize_text = output_visualize_text

class InitSelfplayParameter:
    def __init__(self, 
                selfplay_repeat : int = 500,
                gamma : float = GAMMA,
                train_epoch : int = 200):
        self.selfplay_repeat = selfplay_repeat
        self.gamma = gamma
        self.train_epoch = train_epoch

class TrainParameter:
    def __init__(self, 
                epoch_count : int = 100):
        self.epoch_count = epoch_count


class ExplorationParameter:
    def __init__(self, 
                c_base : float=C_BASE, 
                c_init : float=C_INIT,
                alpha : float=ALPHA,
                epsilon : float=EPSILON,
                temperature: float = TEMPERATURE
                ):
        self.c_base = c_base
        self.c_init = c_init
        self.alpha = alpha
        self.epsilon = epsilon
        self.temperature = temperature


