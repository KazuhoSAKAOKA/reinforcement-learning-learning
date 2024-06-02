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

class HistoryData:
    def __init__(self, path:str):
        self.history = []
        self.path = path
        if os.path.exists(path):
            with open(path, mode='rb') as f:
                self.history = pickle.load(f)
    def extend(self, data_first, data_second):
        self.history.extend(data_first)
        self.history.extend(data_second)
    def serialize(self):
        with open(self.path, mode='wb') as f:
            pickle.dump(self.history, f)
    def get_info(self):
        return self.path
    def get_path_list(self)->list:
        return [self.path]

class DualHistoryData:
    def __init__(self, path_first:str, path_second:str):
        self.history_data_first = HistoryData(path_first)
        self.history_data_second = HistoryData(path_second)
    def extend(self, data_first, data_second):
        self.history_data_first.history.extend(data_first)
        self.history_data_second.history.extend(data_second)
    def serialize(self):
        self.history_data_first.serialize()
        self.history_data_second.serialize()
    def get_info(self):
        return self.history_data_first.path + ',' + self.history_data_second.path
    def get_path_list(self)->list:
        return [self.history_data_first.path, self.history_data_second.path]

def prepare_dir(folder:str)->str:
    if(folder[-1] != '/'):
        folder = '{0}/'.format(folder)
    os.makedirs(folder, exist_ok=True)
    return folder

def init_history_data(param:SelfplayParameter)->HistoryData:
    now = datetime.now()
    if param.history_folder_second is None or param.history_folder_second == '':
        path = prepare_dir(param.history_folder) + '{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        return HistoryData(path)
    else:
        filename = '{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        path1 = prepare_dir(param.history_folder) + filename
        path2 = prepare_dir(param.history_folder_second) + filename
        return DualHistoryData(path1, path2)

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
            selfplay_param: SelfplayParameter) -> list[str]:
    env = GameEnv(game_board=game_board, first_agent=Agent(first_brain),second_agent= Agent(second_brain))
    first_win = 0
    second_win = 0
    draw = 0
    if selfplay_param.is_continue:
        first_history_path = sorted(Path(selfplay_param.history_folder).glob('*.history'))[-1]
        if selfplay_param.history_folder_second is not None and selfplay_param.history_folder_second != '':
            second_history_path = sorted(Path(selfplay_param.history_folder_second).glob('*.history'))[-1]
            history_data = DualHistoryData(first_history_path,second_history_path)
        else:
            history_data = HistoryData(first_history_path)
    else:
        history_data = init_history_data(selfplay_param)
    print('Self play start. start:{}, history_file:{}'.format(datetime.now(), history_data.get_info()))

    for i in range(selfplay_param.selfplay_repeat):
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

        history_data.extend(game_board.augmente_data(first_brain.history), game_board.augmente_data(second_brain.history))
        history_data.serialize()

        first_brain.reset()
        second_brain.reset()

        stop = datetime.now()
        print('\rSelf play {}/{} start={},stop={}, duration={}'.format(i + 1, selfplay_param.selfplay_repeat, start, stop, stop-start))

    print('\rcomplete. self play first_win:{} second_win:{} draw:{}'.format(first_win, second_win, draw))
    print('history file:{}'.format(history_data))
    return history_data.get_path_list()



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
            history_folder_second: str,
            is_continue:bool = False,
            start_index:int = 0) -> Tuple[str,str]:
    param = SelfplayParameter(
        history_folder=history_folder_first,
        history_folder_second=history_folder_second,
        selfplay_repeat=repeat_count,
        is_continue=is_continue,
        start_index=start_index
    )
    l = self_play_impl(first_brain, second_brain, game_board, param)
    return l[0], l[1]



