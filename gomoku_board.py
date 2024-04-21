from typing import Tuple
import numpy as np
from game_board import GameBoard, GameRelativeResult, GameResult

class StoneSeries:
    def __init__(self, connected_count):
        self.connected_count_without_skip = connected_count
        self.connected_count = connected_count
        self.front_edge_open = True
        self.back_edge_open = True
        self.skipped = False
        self.skip_direction = None
    def get_connected(self):
        return self.connected_count
    def get_connected_without_skip(self):
        return self.connected_count_without_skip
    def get_front_edge_open(self):
        return self.front_edge_open
    def get_back_edge_open(self):
        return self.back_edge_open
    def get_skipped(self):
        return self.skipped
    def extend(self, to_front):
        self.connected_count += 1
        if self.skipped and self.skip_direction == to_front:
            pass
        else:
            self.connected_count_without_skip += 1
    def set_skip(self, to_front):
        if self.skipped:
            Exception("Already skipped")
        self.skipped = True
        self.skip_direction = to_front
    def close_front(self):
        self.front_edge_open = False
    def close_back(self):
        self.back_edge_open = False


directions = [
    (1, 0),
    (1, 1),
    (-1, 1),
    (0, 1),
]

class GomokuBoard(GameBoard):
    def __init__(self, board_size = 15, turn = 0, last_action = None):
        super().__init__(board_size, turn, last_action)

    def get_model_state(self):
        return np.array([self.self_cells, self.enemy_cells])

    def counting_connected(self, judge_cells, other_cells, x, y, dx, dy, to_front, series)->StoneSeries:
        '''
        接続している石の数を数える。
        とび石がある場合は、それを考慮する。
        '''
        #　盤外の場合
        if x < 0 or y < 0 or x >= self.board_size or y >= self.board_size:
            if to_front:
                series.close_front()
            else:
                series.close_back()
            return series
        # 相手の石がある場合
        if other_cells[y][x] == 1:
            if to_front:
                series.close_front()
            else:
                series.close_back()
            return series
        # 石がない場合
        if judge_cells[y][x] == 0:
            # 飛び石は一つまで考慮する
            if series.get_skipped():
                return series
            else:
                series.set_skip(to_front)
        else:
            series.extend(to_front)
        return self.counting_connected(judge_cells, other_cells, x + dx, y + dy, dx, dy, to_front, series)
    
    def get_line_series(self, judge_cells, other_cells, x, y, direction)->StoneSeries:
        '''
        指定した石の指定した方向への並びの種類、ここでは５連や４連等の情報を返す
        '''
        if direction < 0 or direction >= 4:
            Exception("Invalid direction")
        if judge_cells[y][x] == 0:
            return StoneSeries(0)
        series = StoneSeries(1)
        dx, dy = directions[direction]
        series = self.counting_connected(judge_cells, other_cells, x + dx, y + dy, dx, dy, True, series)
        series = self.counting_connected(judge_cells, other_cells, x - dx, y - dy, -dx, -dy, False, series)
        return series
    
    def judge_last_action(self)->Tuple[bool, GameRelativeResult]:
        '''
        最後に打った手による勝敗を判定する

        '''
        if self.last_action is None:
            return False, None
        x = self.last_action % self.board_size
        y = self.last_action // self.board_size
        active_threes = 0
        active_fours = 0
        for i in range(4):
            series = self.get_line_series(self.enemy_cells, self.self_cells, x, y, i)
            if series.get_connected_without_skip() == 5:
                return True, GameRelativeResult.win_last_play_player
            elif series.get_connected_without_skip() > 5:
                # ６個以上並んでいる場合、先手なら負け、後手なら勝ち
                if self.is_first_player_last_operated():
                    return True, GameRelativeResult.lose_last_play_player
                else:
                    return True, GameRelativeResult.win_last_play_player
            if series.get_connected() == 3:
                if series.get_front_edge_open() and series.get_back_edge_open():
                    active_threes += 1
            elif series.get_connected() == 4:
                if series.get_front_edge_open() or series.get_back_edge_open():
                    active_fours += 1
        
        # 先手であった場合、三々、四々は禁じ手
        if self.is_first_player_last_operated():
            if active_threes >= 2 or active_fours >= 2:
                return True, GameRelativeResult.lose_last_play_player

        if self.turn == self.board_size * self.board_size:
            return True, GameRelativeResult.draw
        return False, None


    def get_legal_actions(self)->np.ndarray:
        actions = np.array([], dtype=np.int8)

        # 初手の場合、盤面の中央に打つ
        if self.turn == 0:
            actions = np.append(actions, (self.board_size // 2) * self.board_size + (self.board_size // 2))
            return actions

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.self_cells[i][j] == 0 and self.enemy_cells[i][j] == 0:
                    actions = np.append(actions, i * self.board_size + j)
        return actions
    def get_legal_actions_ratio(self)->np.ndarray:
        '''
        行動可能なアクションを1.0,不可能なアクションを0.0とした配列を返す
        '''
        actions = np.zeros(self.board_size * self.board_size, dtype=np.float32)

        # 初手の場合、盤面の中央に打つ
        if self.turn == 0:
            actions[self.board_size//2][self.board_size//2] = 1.0
            return 
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.self_cells[i][j] == 0 and self.enemy_cells[i][j] == 0:
                    actions[i][j] = 1.0
        return actions
def test_board():
    board = board = GomokuBoard(7)
    while True:
        board.output_for_debug()
        if board.is_first_player_turn():
            print("First player's turn")
        else:
            print("Second player's turn")
        actions = board.get_legal_actions()
        print(actions)
        while True:
            action = int(input())
            if action in actions:
                break
            print("Invalid action")
        _, board = board.transit_next(action)
        done, result = board.judge_last_action()
        if done:
            print(result)
            break


if __name__ == '__main__':
    test_board()

