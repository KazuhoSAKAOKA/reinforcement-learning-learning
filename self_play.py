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

class HistoryData:
    def __init__(self, path:str):
        self.history = []
        self.path = path
        if os.path.exists(path):
            with open(path, mode='rb') as f:
                self.history = pickle.load(f)
    def extend(self, data):
        self.history.extend(data)
    def serialize(self):
        with open(self.path, mode='wb') as f:
            pickle.dump(self.history, f)
    def get_path(self):
        return self.path
    def get_history(self):
        return self.history

def prepare_dir(folder:str)->str:
    if(folder[-1] != '/'):
        folder = '{0}/'.format(folder)
    os.makedirs(folder, exist_ok=True)
    return folder

def init_history_data(folder:str)->HistoryData:
    now = datetime.now()
    path =  prepare_dir(folder) + '{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    return HistoryData(path)

def init_history_data_dual(folder1:str, folder2:str)->Tuple[HistoryData, HistoryData]:
    now = datetime.now()
    filename = '{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    path1 = prepare_dir(folder1) + filename
    path2 = prepare_dir(folder2) + filename
    return HistoryData(path1), HistoryData(path2)

#def write_data(folder:str, history)->str:
#    now = datetime.now()
#    if(folder[-1] != '/'):
#        folder = '{0}/'.format(folder)
#    os.makedirs(folder, exist_ok=True)
#    path =  folder + '{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
#    with open(path, mode='wb') as f:
#        pickle.dump(history, f)
#    return path

def load_data_file(history_file : Path):
    with history_file.open(mode='rb') as f:
        return pickle.load(f)
def load_data(history_folder):
    history_path = sorted(Path(history_folder).glob('*.history'))[-1]
    return load_data_file(history_path)
def load_data_file_name(history_file : str):
    return load_data_file(Path(history_file))



def self_play(
            first_brain: SelfplayBrain, 
            second_brain : SelfplayBrain, 
            game_board : GameBoard, 
            repeat_count : int, 
            history_folder: str,
            is_continue:bool = False,
            start_index:int = 0
            ) -> str:
    
    env = GameEnv(game_board=game_board, first_agent=Agent(first_brain),second_agent= Agent(second_brain))
    first_win = 0
    second_win = 0
    draw = 0
    if is_continue:
        history_path = sorted(Path(history_folder).glob('*.history'))[-1]
        history_data = HistoryData(history_path)
    else:
        history_data = init_history_data(history_folder)
    print('Self play start. start:{}, history_file:{}'.format(datetime.now(), history_data.get_path()))

    for i in range(start_index, repeat_count):
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
        history_data.extend(game_board.augmente_data(first_brain.history))
        history_data.extend(game_board.augmente_data(second_brain.history))
        history_data.serialize()
        first_brain.reset()
        second_brain.reset()

        stop = datetime.now()
        print('\rSelf play {}/{} start={},stop={}, duration={}'.format(i + 1, repeat_count, start,stop, stop-start))
    print('\rcomplete. self play first_win:{} second_win:{} draw:{}'.format(first_win, second_win, draw))
    print('history file:{}'.format(history_data.get_path()))
    return history_data.get_path()

def self_play_dualmodel(
            first_brain: SelfplayBrain, 
            second_brain : SelfplayBrain, 
            game_board : GameBoard, 
            repeat_count : int, 
            history_folder_first: str, 
            history_folder_second: str,
            is_continue:bool = False,
            start_index:int = 0) -> Tuple[str,str]:
    env = GameEnv(game_board=game_board, first_agent=Agent(first_brain),second_agent= Agent(second_brain))
    first_win = 0
    second_win = 0
    draw = 0
    if is_continue:
        first_history_path = sorted(Path(history_folder_first).glob('*.history'))[-1]
        first_history_data = HistoryData(first_history_path)
        second_history_path = sorted(Path(history_folder_second).glob('*.history'))[-1]
        second_history_data = HistoryData(second_history_path)
    else:
        first_history_data,second_history_data = init_history_data_dual(history_folder_first, history_folder_second)
    print('Self play start. start:{}, history_file:{},history_file:{}'.format(datetime.now(), first_history_data.get_path(), second_history_data.get_path()))

    for i in range(repeat_count):
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
        first_history_data.extend(game_board.augmente_data(first_brain.history))
        second_history_data.extend(game_board.augmente_data(second_brain.history))
        first_history_data.serialize()
        second_history_data.serialize()
        first_brain.reset()
        second_brain.reset()

        stop = datetime.now()
        print('\rSelf play {}/{} start={},stop={}, duration={}'.format(i + 1, repeat_count, start,stop, stop-start))

    print('\rcomplete. self play first_win:{} second_win:{} draw:{}'.format(first_win, second_win, draw))
    print('history file:{}, {}'.format(first_history_data.get_path(), second_history_data.get_path()))
    return first_history_data.get_path(),second_history_data.get_path()



