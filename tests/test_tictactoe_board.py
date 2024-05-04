import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from game_board import GameBoard, GameRelativeResult, GameResult
from stone_game_board import StoneGameBoard
from tictactoe_board import TicTacToeBoard


class TestTicTacToeBoard(unittest.TestCase):

    def do_step_setup(self, board : TicTacToeBoard, x : int, y : int)->TicTacToeBoard:
        board, r = board.transit_next(board.xy_to_index(x, y))
        if not r:
            print(board)
        self.assertTrue(r)
        done, _ = board.judge_last_action()
        if done:
            print(board)
        self.assertFalse(done)
        return board

    def do_step_last(self, board : TicTacToeBoard, x : int, y : int)->TicTacToeBoard:
        board, r = board.transit_next(board.xy_to_index(x, y))
        if not r:
            print(board)
        self.assertTrue(r)
        return board
    def do_steps(self, board : TicTacToeBoard, stones :  list)->TicTacToeBoard:
        for x, y in stones[:-1]:
            board = self.do_step_setup(board, x, y)
        last_x, last_y = stones[-1]
        board = self.do_step_last(board, last_x, last_y)
        return board

    def test_simple_win_first_player(self):
        board = TicTacToeBoard()
        stones = [(0,0), (1,0), (1,1), (2,0), (2,2)]
        board = self.do_steps(board=board, stones=stones)      
        done, result = board.judge_last_action()
        self.assertTrue(done)
        self.assertEqual(result, GameRelativeResult.win_last_play_player)
        self.assertEqual(board.convert_to_result(result), GameResult.win_first_player)
    def test_simple_win_second_player(self):
        board = TicTacToeBoard()
        stones = [(0,0), (1,1), (0,1), (2,0), (2,2), (0,2)]
        board = self.do_steps(board=board, stones=stones)      
        done, result = board.judge_last_action()
        self.assertTrue(done)
        self.assertEqual(result, GameRelativeResult.win_last_play_player)
        self.assertEqual(board.convert_to_result(result), GameResult.win_second_player)

    def test_simple_draw(self):
        board = TicTacToeBoard()
        stones = [(0,0), (1,0), (2,0), (1,1), (0,1), (2,1), (1,2), (0,2), (2,2)]
        board = self.do_steps(board=board, stones=stones)      
        done, result = board.judge_last_action()
        self.assertTrue(done)
        self.assertEqual(result, GameRelativeResult.draw)
        self.assertEqual(board.convert_to_result(result), GameResult.draw)