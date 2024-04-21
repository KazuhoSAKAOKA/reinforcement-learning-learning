from typing import Tuple
from agent import Agent
from game_board import GameBoard, get_first_player_value, GameResult
from game import GameEnv
from datetime import datetime
from selfplay_brain import SelfplayBrain, SelfplayRandomBrain
import pickle
import os
from tictactoe_board import TicTacToeBoard
from pathlib import Path

def write_data(folder:str, history)->str:
    now = datetime.now()
    folder = '{0}/'.format(folder)
    os.makedirs(folder, exist_ok=True)
    path = folder + '/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)
    return path

def load_data_file(history_file : Path):
    with history_file.open(mode='rb') as f:
        return pickle.load(f)
def load_data(history_folder):
    history_path = sorted(Path(history_folder).glob('*.history'))[-1]
    return load_data_file(history_path)
def load_data_file_name(history_file : str):
    return load_data_file(Path(history_file))



def self_play(first_brain: SelfplayBrain, second_brain : SelfplayBrain, board : GameBoard, repeat_count : int, history_folder: str) -> str:
    history=[]
    env = GameEnv(board, Agent(first_brain), Agent(second_brain))
    first_win = 0
    second_win = 0
    draw = 0
    for i in range(repeat_count):
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
        history.extend(first_brain.history)
        history.extend(second_brain.history)
        #history.append([board.get_model_state(), [0] * board.get_output_size(), value])
        first_brain.reset()
        second_brain.reset()
        print('\rSelf play {}/{}'.format(i + 1, repeat_count), end='')
    filename = write_data(history_folder, history)
    print('\rcomplete. self play first_win:{} second_win:{} draw:{}'.format(first_win, second_win, draw))
    print('history file:{}'.format(filename))
    return filename

def self_play2(first_brain: SelfplayBrain, second_brain : SelfplayBrain, board : GameBoard, repeat_count : int, history_folder_first: str, history_folder_second: str) -> Tuple[str,str]:
    history_first=[]
    history_second=[]
    env = GameEnv(board, Agent(first_brain), Agent(second_brain))
    first_win = 0
    second_win = 0
    draw = 0
    for i in range(repeat_count):
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
        history_first.extend(first_brain.history)
        history_second.extend(second_brain.history)
        #history.append([board.get_model_state(), [0] * board.get_output_size(), value])
        first_brain.reset()
        second_brain.reset()
        print('\rSelf play {}/{}'.format(i + 1, repeat_count), end='')
    filename_first = write_data(history_folder_first, history_first)
    filename_second = write_data(history_folder_second, history_second)
    print('\rcomplete. self play first_win:{} second_win:{} draw:{}'.format(first_win, second_win, draw))
    print('history file:{}, {}'.format(filename_first, filename_second))
    return filename_first,filename_second


def debug_tictactoe():
    board = TicTacToeBoard()
    first_brain = SelfplayRandomBrain()
    second_brain = SelfplayRandomBrain()
    count = 2000
    self_play(first_brain, second_brain, board, count, './data/tictactoe')
    self_play2(first_brain, second_brain, board, count, './data/tictactoe/first','./data/tictactoe/second')


if __name__ == '__main__':
    debug_tictactoe()

