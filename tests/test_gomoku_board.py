import unittest

from gomoku_board import GomokuBoard, GameRelativeResult

class TestGomokuBoard(unittest.TestCase):

    def do_step_setup(self, board : GomokuBoard, x : int, y : int)->GomokuBoard:
        board, r = board.transit_next(board.xy_to_index(x, y))
        if not r:
            print(board)
        self.assertTrue(r)
        done, _ = board.judge_last_action()
        if done:
            print(board)
        self.assertFalse(done)
        return board

    def do_step_last(self, board : GomokuBoard, x : int, y : int)->GomokuBoard:
        board, r = board.transit_next(board.xy_to_index(x, y))
        if not r:
            print(board)
        self.assertTrue(r)
        return board


    def do_steps(self, board : GomokuBoard, stones :  list)->GomokuBoard:
        for x, y in stones[:-1]:
            board = self.do_step_setup(board, x, y)
        last_x, last_y = stones[-1]
        board = self.do_step_last(board, last_x, last_y)
        return board

    def test_simple_horizontal5(self):
        board = GomokuBoard(15)
        stones = [(7,7), (6,6), (6,7), (5,5), (8,7), (4,4), (5,7), (3,3), (9, 7)]
        board = self.do_steps(board=board, stones=stones)      
        done, result = board.judge_last_action()
        self.assertTrue(done)
        self.assertEqual(result, GameRelativeResult.win_last_play_player)

    def test_simple_vertical5(self):
        board = GomokuBoard(15)
        stones = [(7,7), (6,6), (7,8), (5,5), (7,6), (4,4), (7,9), (3,3), (7, 5) ]
        board = self.do_steps(board=board, stones=stones)      
        done, result = board.judge_last_action()
        self.assertTrue(done)
        self.assertEqual(result, GameRelativeResult.win_last_play_player)

    def test_simple_slant5(self):
        board = GomokuBoard(15)
        stones = [(7,7), (5,6), (6,6), (4,5), (5,5), (3,4), (4,4), (2,3), (3,3) ]
        board = self.do_steps(board=board, stones=stones)
        done, result = board.judge_last_action()
        if not done:
            print(board)
        self.assertTrue(done)
        self.assertEqual(result, GameRelativeResult.win_last_play_player)
        board = GomokuBoard(15)
        stones = [(7,7), (5,6), (8,6), (4,5), (9,5), (3,4), (10,4), (2,3), (11,3) ]
        board = self.do_steps(board=board, stones=stones)
        done, result = board.judge_last_action()
        self.assertTrue(done)
        self.assertEqual(result, GameRelativeResult.win_last_play_player)

    def test_three_three_lose(self):
        board = GomokuBoard(15)
        stones = [(7,7), (5,6), (8,6), (4,5), (9,7), (3,4), (9,8), (2,3), (9,5) ]
        board = self.do_steps(board=board, stones=stones)
        done, result = board.judge_last_action()
        if not done:
            print(board)
        self.assertTrue(done)
        self.assertEqual(result, GameRelativeResult.lose_last_play_player)
