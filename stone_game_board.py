from typing import Tuple
from game_board import GameBoard, GameRelativeResult, GameResult
import numpy as np

class StoneGameBoard(GameBoard):
    def __init__(self, board_size : int, turn: int = 0 , last_action : int = -1):
        super().__init__(turn, last_action)
        self.self_cells = np.zeros((board_size, board_size), dtype=np.int8)
        self.enemy_cells = np.zeros((board_size, board_size), dtype=np.int8)
        self.board_size : int = board_size
    def reset(self):
        self.self_cells = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.enemy_cells = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        super().reset()

    def get_output_size(self):
        return self.board_size * self.board_size

    def get_cell(self, x, y):
        if self.self_cells[y][x] == 1:
            return 1 if self.is_first_player_turn() else 2
        if self.enemy_cells[y][x] == 1:
            return 2 if self.is_first_player_turn() else 1
        return 0
    def to_hisotry_record(self)->list:
        x = [self.self_cells.reshape(self.board_size*self.board_size,).tolist(), self.enemy_cells.reshape(self.board_size*self.board_size,).tolist()]
        return x

    def reshape_to_input(self)->np.ndarray:
        x = np.reshape([self.self_cells, self.enemy_cells], (2, self.board_size, self.board_size))
        x = x.transpose(1, 2, 0)
        x = x.reshape(1, self.board_size, self.board_size, 2)
        return x
    def index_to_xy(self, index:int)->Tuple[int,int]:
        return index % self.board_size, index // self.board_size
    def xy_to_index(self, x:int, y:int)->int:
        return y * self.board_size + x
    
    def reshape_history_to_input(self, history) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, y_policies, y_values = zip(*history)
        xs = np.array(xs)
        a,b,c = (self.board_size, self.board_size, 2)
        xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
        y_policies = np.array(y_policies)
        y_values = np.array(y_values)
        return xs, y_policies, y_values

    # historyから状態を復元
    def deserialize(self, history) -> list:
        states = []
        for (x, y_policy, y_value) in history:
            game_board = StoneGameBoard(self.board_size)
            game_board.self_cells = np.array(x[0]).reshape(game_board.board_size, game_board.board_size)
            game_board.enemy_cells = np.array(x[1]).reshape(game_board.board_size, game_board.board_size)
            game_board.turn = np.sum(x[0][:]) + np.sum(x[1][:])
            states.append((game_board, y_policy, y_value))
        return states
    
    def transit_next(self, action)-> Tuple['StoneGameBoard', bool]:
        '''
        次の状態を返す
        '''
        x = action % self.board_size
        y = action // self.board_size
        if self.self_cells[y][x] != 0 or self.enemy_cells[y][x] != 0:
            return self, False
        next = self.__class__()
        next.self_cells = self.enemy_cells.copy()
        next.enemy_cells = self.self_cells.copy()
        next.enemy_cells[y][x] = 1
        next.board_size = self.board_size
        next.turn = self.turn + 1
        next.last_action = action
        return next, True
    
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

    def get_legal_actions(self)->np.ndarray:
        '''
        空いているマスのリストを返す
        '''
        actions = np.array([], dtype=np.int8)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.self_cells[i][j] == 0 and self.enemy_cells[i][j] == 0:
                    actions = np.append(actions, i * self.board_size + j)
        return actions

    def to_state_key(self)->str:
        temp = ''
        if self.is_first_player_turn():
            my = 'o'
            you = 'x'
        else:
            my = 'x'
            you = 'o'
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.self_cells[i][j] == 1:
                    temp += my
                elif self.enemy_cells[i][j] == 1:
                    temp += you
                else:
                    temp += '-'
        return temp

    def augmente_data(self, history) -> any:
        new_history = history.copy()
        for x, y_policy, y_value in history:
            x = np.array(x)
            y_policy = np.array(y_policy)

            x1 = x[0][:].reshape([self.board_size, self.board_size])
            x2 = x[1][:].reshape([self.board_size, self.board_size])
            y_policy = y_policy.reshape([self.board_size, self.board_size])

            for i in range(3):
                x1 = np.rot90(x1)
                x2 = np.rot90(x2)
                y_policy = np.rot90(y_policy)
                xhis = [x1.reshape(self.board_size*self.board_size,).tolist(), x2.reshape(self.board_size*self.board_size,).tolist()]
                y_policy_his = y_policy.reshape(self.board_size*self.board_size,).tolist()
                new_history.append((xhis, y_policy_his, y_value))

            x1 = np.fliplr(x1)
            x2 = np.fliplr(x2)
            y_policy = np.fliplr(y_policy)
            xhis = [x1.reshape(self.board_size*self.board_size,).tolist(), x2.reshape(self.board_size*self.board_size,).tolist()]
            y_policy_his = y_policy.reshape(self.board_size*self.board_size,).tolist()
            new_history.append((xhis, y_policy_his, y_value))
            for i in range(3):
                x1 = np.rot90(x1)
                x2 = np.rot90(x2)
                y_policy = np.rot90(y_policy)
                xhis = [x1.reshape(self.board_size*self.board_size,).tolist(), x2.reshape(self.board_size*self.board_size,).tolist()]
                y_policy_his = y_policy.reshape(self.board_size*self.board_size,).tolist()
                new_history.append((xhis, y_policy_his, y_value))
        return new_history

    def history_row_to_info(self, row:np.ndarray)->Tuple['StoneGameBoard', np.ndarray, float]:
        x = row[0].reshape(self.board_size, self.board_size)
        game_board = self.__class__(self.board_size)
        game_board.self_cells = np.array(x[0]).reshape(game_board.board_size, game_board.board_size)
        game_board.enemy_cells = np.array(x[1]).reshape(game_board.board_size, game_board.board_size)
        game_board.turn = np.sum(x[0][:]) + np.sum(x[1][:])
        y_policy = row[1].reshape(self.board_size, self.board_size)
        y_value = row[2]
        return x, y_policy, y_value
    
