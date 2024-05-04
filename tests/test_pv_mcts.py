from typing import Tuple
import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from game_board import GameBoard
from tictactoe_board import TicTacToeBoard
from pv_mcts import pv_mcts_policies
from parameter import PARAM

import numpy as np

class Test_pv_mcts(unittest.TestCase):
    def predict_test_same(board: GameBoard)->Tuple[np.ndarray, float]:
        count = len(board.get_legal_actions())
        return (board.get_legal_actions_ratio() / count), 0.0

    def assertArrayLen(self, scores : list, begin:int, end: int, expect: float):
        for i in range(begin, min(end, len(scores))):
            self.assertAlmostEqual(scores[i], expect)
    def assertArrayRemain(self, scores : list, begin:int, expect: float):
        self.assertArrayLen(scores, begin, len(scores), expect)

    # ボルツマン温度,ディリクレノイズがない場合の挙動確認
    def test_pv_mcts(self):
        PARAM.alpha = 0.0
        PARAM.epsilon = 0.0
        PARAM.temperature = 0.0
        board = TicTacToeBoard()

        scores = pv_mcts_policies(board, 0, Test_pv_mcts.predict_test_same, Test_pv_mcts.predict_test_same)
        self.assertArrayRemain(scores, 0, 1.0 / 9)

        # すべての手が同じ評価値。1週目
        for i in range(1, 9):
            scores = pv_mcts_policies(board, i, Test_pv_mcts.predict_test_same, Test_pv_mcts.predict_test_same)
            self.assertEqual(len(scores), 9)
            if i < 9:
                self.assertArrayRemain(scores, i + 1, 0.0)
            if i > 0:
                self.assertArrayLen(scores, 0, i, 1.0 / i)
        # 評価回数が増えた場合
        for i in range(10, 17):
            scores = pv_mcts_policies(board, i, Test_pv_mcts.predict_test_same, Test_pv_mcts.predict_test_same)
            self.assertEqual(len(scores), 9)
            self.assertArrayLen(scores, 0, i - 9, 1.0 / i * 2)
            self.assertArrayRemain(scores, i - 9, 1.0 / i)
        
    def test_pv_mcts_progress(self):
        PARAM.alpha = 0.0
        PARAM.epsilon = 0.0
        board = TicTacToeBoard()
        board, r = board.transit_next(0)
        self.assertTrue(r)
        scores = pv_mcts_policies(board, 8, Test_pv_mcts.predict_test_same, Test_pv_mcts.predict_test_same)
        self.assertEqual(len(scores), 9)
        self.assertEqual(scores[0], 0.0)
        self.assertArrayRemain(scores, 1, 1.0 / 8)

        board, r = board.transit_next(1)
        self.assertTrue(r)
        scores = pv_mcts_policies(board, 7, Test_pv_mcts.predict_test_same, Test_pv_mcts.predict_test_same)
        self.assertEqual(len(scores), 9)
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)
        self.assertArrayRemain(scores, 2, 1.0 / 7)

        board, r = board.transit_next(2)
        self.assertTrue(r)
        board, r = board.transit_next(4)
        self.assertTrue(r)
        board, r = board.transit_next(3)
        self.assertTrue(r)
        board, r = board.transit_next(5)
        self.assertTrue(r)
        board, r = board.transit_next(7)
        self.assertTrue(r)
        board, r = board.transit_next(6)
        self.assertTrue(r)

        scores = pv_mcts_policies(board, 100, Test_pv_mcts.predict_test_same, Test_pv_mcts.predict_test_same)
        self.assertEqual(len(scores), 9)
        self.assertArrayLen(scores, 0, 8, 0.0)
        self.assertArrayRemain(scores, 8, 1.0)

