from typing import Tuple
from agent import Agent
from game_board import GameBoard, get_first_player_value, GameResult
from game import GameEnv
from datetime import datetime
from self_play_brain import SelfplayBrain
import pickle
import os
from tictactoe_board import TicTacToeBoard
from pathlib import Path
from parameter import SelfplayParameter


class HistoryDataBase:

    #abstractmethod
    def get_count(self)->int:
        pass
    #abstractmethod
    def serialize(self, data_1 : list, data_2 : list):
        pass
    #abstractmethod
    def get_primary(self)->'HistoryDataBase':
        pass
    #abstractmethod
    def get_secondary(self)->'HistoryDataBase':
        pass
    #abstractmethod
    def deserialize(self)->list:
        pass

class HistoryData(HistoryDataBase):
    def __init__(self, folder_path:str):
        self.folder = Path(folder_path)
        if os.path.exists(folder_path) and (not os.path.isdir(folder_path)):
            self.count = len(self.folder.glob('*.history'))
        else:
            os.makedirs(folder_path, exist_ok=True)
            self.count = 0
    def get_count(self)->int:
        return self.count
    def serialize(self, data_1 : list, data_2 : list):
        history = []
        if data_1 is not None:
            history.extend(data_1)
        if data_2 is not None:
            history.extend(data_2)
        self.path = self.folder / '{:04}.history'.format(self.count)
        with open(self.path, mode='wb') as f:
            pickle.dump(history, f)
        self.count += 1

    def get_primary(self)->'HistoryDataBase':
        return self
    def get_secondary(self)->'HistoryDataBase':
        return None

    def deserialize(self)->list:
        files = self.folder.glob('*.history')
        history = []
        for file in files:
            with file.open(mode='rb') as f:
                history.extend(pickle.load(f))
        return history
    
    def __repr__(self) -> str:
        return '{}'.format(self.folder)
class DualHistoryData(HistoryDataBase):
    def __init__(self, folder_path:str, first_key:str='first', second_key:str='second'):
        self.folder = Path(folder_path)
        path_first = self.folder / first_key
        path_second = self.folder / second_key
        self.history_data_first = HistoryData(path_first)
        self.history_data_second = HistoryData(path_second)
    def serialize(self, data_1, data_2):
        self.history_data_first.serialize(data_1, None)
        self.history_data_second.serialize(data_2, None)

    def get_count(self)->int:
        return self.history_data_first.get_count()
    

    def get_primary(self)->'HistoryDataBase':
        return self.history_data_first
    def get_secondary(self)->'HistoryDataBase':
        return self.history_data_second

    def __repr__(self) -> str:
        return '{},{}'.format(self.history_data_first, self.history_data_second)


def prepare_dir(folder:str)->str:
    if(folder[-1] != '/'):
        folder = '{0}/'.format(folder)
    os.makedirs(folder, exist_ok=True)
    return folder

def init_history_data(is_dual_model: bool, selfplay_param:SelfplayParameter)->HistoryData:
    now = datetime.now()
    path = prepare_dir(selfplay_param.history_folder) + '{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    if not is_dual_model:
        return HistoryData(path)
    else:
        return DualHistoryData(path)

def load_data_file(history_file : Path):
    with history_file.open(mode='rb') as f:
        return pickle.load(f)
def load_data(history_folder):
    history_path = sorted(Path(history_folder).glob('*.history'))[-1]
    return load_data_file(history_path)
def load_data_file_name(history_file : str):
    return load_data_file(Path(history_file))



def self_play_impl(
            first_brain: SelfplayBrain, 
            second_brain : SelfplayBrain, 
            game_board : GameBoard, 
            is_dual_model: bool,
            selfplay_param: SelfplayParameter) -> HistoryDataBase:
    env = GameEnv(game_board=game_board, first_agent=Agent(first_brain),second_agent= Agent(second_brain))
    first_win = 0
    second_win = 0
    draw = 0
    if selfplay_param.continue_history_folder_path is not None:
        if selfplay_param.is_dual_model:
            history_data = DualHistoryData(selfplay_param.continue_history_folder_path)
        else:
            history_data = HistoryData(selfplay_param.continue_history_folder_path)
    else:
        history_data = init_history_data(is_dual_model=is_dual_model,selfplay_param=selfplay_param)
    print('Self play start. start:{}, history_file:{}'.format(datetime.now(), history_data))
    
    while history_data.get_count() < selfplay_param.selfplay_repeat:
        start = datetime.now()
        result = env.play()
        if result == GameResult.win_first_player:
            first_win += 1
        elif result == GameResult.win_second_player:
            second_win += 1
        else:
            draw += 1
        value = get_first_player_value(result)
        first_brain.update_history(value)
        second_brain.update_history(-value)

        history_data.serialize(first_brain.history, second_brain.history)

        first_brain.reset()
        second_brain.reset()

        stop = datetime.now()
        print('\rSelf play {}/{} start={},stop={}, duration={}'.format(history_data.get_count(), selfplay_param.selfplay_repeat, start, stop, stop-start))

    print('\rcomplete. self play first_win:{} second_win:{} draw:{}'.format(first_win, second_win, draw))
    print('history file:{}'.format(history_data))
    return history_data


'''
def self_play(
            first_brain: SelfplayBrain, 
            second_brain : SelfplayBrain, 
            game_board : GameBoard, 
            repeat_count : int, 
            history_folder: str,
            is_continue:bool = False,
            start_index:int = 0
            ) -> str:
    param = SelfplayParameter(
        history_folder=history_folder,
        selfplay_repeat=repeat_count,
        is_continue=is_continue,
        start_index=start_index)
    l = self_play_impl(first_brain, second_brain, game_board, param)
    return l[0]


def self_play_dualmodel(
            first_brain: SelfplayBrain, 
            second_brain : SelfplayBrain, 
            game_board : GameBoard, 
            repeat_count : int, 
            history_folder_first: str, 
            continue_history_folder_path:str=None) -> Tuple[str,str]:
    param = SelfplayParameter(
        history_folder=history_folder_first,
        selfplay_repeat=repeat_count,
        continue_history_folder_path=continue_history_folder_path)
    l = self_play_impl(first_brain, second_brain, game_board, param)
    return l[0], l[1]
'''


