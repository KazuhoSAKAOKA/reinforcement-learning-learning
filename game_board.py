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
    def __init__(self, board_size, turn = 0, last_action = None):
        self.self_cells = np.zeros((board_size, board_size), dtype=np.int8)
        self.enemy_cells = np.zeros((board_size, board_size), dtype=np.int8)
        self.board_size = board_size
        self.turn = turn
        self.last_action = last_action
    def get_turn(self):
        return self.turn
    def reset(self):
        self.self_cells = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.enemy_cells = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.turn = 0
        self.last_action = None
    def get_output_size(self):
        return self.board_size * self.board_size
    def get_cell(self, x, y):
        if self.self_cells[y][x] == 1:
            return 1 if self.is_first_player_turn() else 2
        if self.enemy_cells[y][x] == 1:
            return 2 if self.is_first_player_turn() else 1
        return 0
    def get_model_state(self):
        x = [self.self_cells.reshape(self.board_size*self.board_size,).tolist(), self.enemy_cells.reshape(self.board_size*self.board_size,).tolist()]
        return x

    def get_model_input_shape(self):
        x = np.reshape([self.self_cells, self.enemy_cells], (2, self.board_size, self.board_size))
        x = x.transpose(1, 2, 0)
        x = x.reshape(1, self.board_size, self.board_size, 2)
        return x
    
    def convert_history_to_model_input(self, history):
        xs, y_policies, y_values = zip(*history)
        xs = np.array(xs)
        a,b,c = (self.board_size, self.board_size, 2)
        xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)        
        y_policies = np.array(y_policies)
        y_values = np.array(y_values)
        return xs, y_policies, y_values

    def transit_next(self, action):
        '''
        次の状態を返す
        '''
        x = action % self.board_size
        y = action // self.board_size
        if self.self_cells[y][x] != 0 or self.enemy_cells[y][x] != 0:
            return False, self
        next = self.__class__()
        next.self_cells = self.enemy_cells.copy()
        next.enemy_cells = self.self_cells.copy()
        next.enemy_cells[y][x] = 1
        next.board_size = self.board_size
        next.turn = self.turn + 1
        next.last_action = action
        return True, next

    def is_first_player_turn(self):
        return self.turn % 2 == 0
    def is_second_player_turn(self):
        return not self.is_first_player_turn()
    def is_first_player_last_operated(self):
        return not self.is_first_player_turn()
    
    def convert_to_result(self, relative_result):
        if relative_result == GameRelativeResult.win_last_play_player:
            if self.is_first_player_last_operated():
                return GameResult.win_first_player
            else:
                return GameResult.win_second_player
        elif relative_result == GameRelativeResult.lose_last_play_player:
            if self.is_first_player_last_operated():
                return GameResult.win_second_player
            else:
                return GameResult.win_first_player
        else:
            return GameResult.draw
    def __repr__(self) -> str:
        s = "---------------------------\n"
        self_stone = "o" if self.is_first_player_turn() else "x"
        enemy_stone = "x" if self.is_first_player_turn() else "o"
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.self_cells[i][j] == 1:
                    s += self_stone
                elif self.enemy_cells[i][j] == 1:
                    s += enemy_stone
                else:
                    s += "-"
            s += "\n"
        s += "\n"
        return s
    def output_for_debug(self):
        print("---------------------------")
        self_stone = "o" if self.is_first_player_turn() else "x"
        enemy_stone = "x" if self.is_first_player_turn() else "o"
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.self_cells[i][j] == 1:
                    print(self_stone, end="")
                elif self.enemy_cells[i][j] == 1:
                    print(enemy_stone, end="")
                else:
                    print("-", end="")
            print("") 
    def get_legal_actions(self):
        '''
        空いているマスのリストを返す
        '''
        actions = np.array([], dtype=np.int8)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.self_cells[i][j] == 0 and self.enemy_cells[i][j] == 0:
                    actions = np.append(actions, i * self.board_size + j)
        return actions
    def get_legal_actions_ratio(self):
        '''
        行動可能なアクションを1.0,不可能なアクションを0.0とした配列を返す
        '''
        actions = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.self_cells[i][j] == 0 and self.enemy_cells[i][j] == 0:
                    actions[i][j] = 1.0
        return actions
