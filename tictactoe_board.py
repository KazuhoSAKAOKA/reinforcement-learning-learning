from typing import Tuple
import numpy as np
from game_board import GameBoard, GameRelativeResult, GameResult
from stone_game_board import StoneGameBoard

directions = [
    (1, 0),
    (1, 1),
    (-1, 1),
    (0, 1),
]

class TicTacToeBoard(StoneGameBoard):
    def __init__(self,board_size=3, turn=0, last_action=None):
        super().__init__(board_size, turn, last_action)



    def counting_connected(self, judge_cells, other_cells, x, y, dx, dy):
        # 盤外の場合
        if x < 0 or y < 0 or x >= self.board_size or y >= self.board_size:
            return 0
        # 相手の石がある場合
        if other_cells[y][x] == 1:
            return 0
        # 石がない場合
        if judge_cells[y][x] == 0:
            return 0
        return 1 + self.counting_connected(judge_cells, other_cells, x + dx, y + dy, dx, dy)
    def get_line_series(self, judge_cells, other_cells, x, y, direction):
        if direction < 0 or direction >= 4:
            Exception("Invalid direction")
        if judge_cells[y][x] == 0:
            return 0
        dx, dy = directions[direction]
        return 1 \
            + self.counting_connected(judge_cells, other_cells, x + dx, y + dy, dx, dy) \
            + self.counting_connected(judge_cells, other_cells, x - dx, y - dy, -dx, -dy)

    def judge_last_action(self)->Tuple[bool, GameRelativeResult]:
        '''
        最後に打った手による勝敗を判定する
        '''
        if self.last_action is None or self.last_action < 0:
            return False, None
        x = self.last_action % self.board_size
        y = self.last_action // self.board_size
        for i in range(4):
            series = self.get_line_series(self.enemy_cells, self.self_cells, x, y, i)
            if series == self.board_size:
                return True, GameRelativeResult.win_last_play_player
        if self.turn == self.board_size * self.board_size:
            return True, GameRelativeResult.draw
        return False, None

def test_board():
    board = TicTacToeBoard()
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
        board, _ = board.transit_next(action)
        done, result = board.judge_last_action()
        if done:
            print(result)
            break


if __name__ == '__main__':
    test_board()