from abc import abstractmethod
from typing import Tuple
import numpy as np
from enum import Enum

class GameRelativeResult(Enum):
    win_last_play_player = 1
    lose_last_play_player = 2
    draw = 3

class GameResult(Enum):
    win_first_player = 1
    win_second_player = 2
    draw = 3

def get_first_player_value(result :GameResult) -> int:
    if result == GameResult.win_first_player:
        return 1
    elif result == GameResult.win_second_player:
        return -1
    else:
        return 0
class GameBoard:
    def __init__(self, turn: int = 0 , last_action : int = -1):
        self.turn  : int = turn
        self.last_action  : int = last_action
    def get_turn(self)->int:
        return self.turn
    def get_last_action(self)->int:
        return self.last_action
    def reset(self):
        self.turn = 0
        self.last_action = -1
    @abstractmethod
    # 行動空間サイズ
    def get_output_size(self)->int:
        pass


    # historyに記憶させる形状に変換
    @abstractmethod
    def to_hisotry_record(self)->any:
        pass

    # モデルのに渡す形状に変換
    @abstractmethod
    def reshape_to_input(self)->np.ndarray:
        pass    
    # historyからモデルに渡す形状に変換
    @abstractmethod
    def reshape_history_to_input(self, history) -> any:
        pass

    # 次の状態への遷移
    @abstractmethod
    def transit_next(self, action)-> Tuple['GameBoard', bool]:
        pass
    
    # 先手番かどうか　パスがあるようなゲームであればオーバーライドする
    def is_first_player_turn(self)->bool:
        return self.turn % 2 == 0
    def is_second_player_turn(self)->bool:
        return not self.is_first_player_turn()
    
    def convert_to_result(self, relative_result : GameRelativeResult)->GameResult:
        if relative_result == GameRelativeResult.win_last_play_player:
            if self.is_second_player_turn():
                return GameResult.win_first_player
            else:
                return GameResult.win_second_player
        elif relative_result == GameRelativeResult.lose_last_play_player:
            if self.is_second_player_turn():
                return GameResult.win_second_player
            else:
                return GameResult.win_first_player
        else:
            return GameResult.draw
    
    def __repr__(self) -> str:
        return "GameBoard"

    def output_for_debug(self):
        print(self)

    @abstractmethod
    def get_legal_actions(self)->np.ndarray:
        pass
    def get_legal_actions_ratio(self)->np.ndarray:
        actions = np.zeros(self.get_output_size(), dtype=np.float32)
        for action in self.get_legal_actions():
            actions[action] = 1.0
        return actions

    @abstractmethod
    def judge_last_action(self)-> Tuple[bool, GameRelativeResult]:
        pass
    def is_done(self)->bool:
        done, _ = self.judge_last_action()
        return done
    