from typing import Tuple
import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from game_board import GameBoard,GameRelativeResult
from tictactoe_board import TicTacToeBoard
from mcts_node import AbstractMctsNode, MctsNode,RandomMctsNode,PolicyValueNetworkMctsNode,UCB1NextNodeSelector,PolicyUCTNextNodeSelector,WithDirichletPolicyUCTNextNodeSelector
from predictor import Predictor,Prediction,DualNetworkPrediction,PolicyNetworkPrediction
from parameter import ExplorationParameter
from threadsafe_dict import ThreadSafeDict

import numpy as np

class MockNode(AbstractMctsNode):
    def __init__(self, selected_count:int, value:float, policy:float):
        self.selected_count = selected_count
        self.value = value
        self.policy = policy
    def get_selected_count(self)->int:
        return self.selected_count
    def get_value(self)->float:
        return self.value
    def get_policy(self)->float:
        return self.policy
    
class Test_UCB1NextNodeSelector(unittest.TestCase):
    def test_select_zero_node(self):
        nodes = list()
        nodes.append(MockNode(10, 0.0, 0.0))
        nodes.append(MockNode(0, 1.0, 0.0))
        nodes.append(MockNode(10, 0.0, 0.0))
        nodes.append(MockNode(10, 0.0, 0.0))
        selector = UCB1NextNodeSelector()
        result1 = selector.select(nodes)
        self.assertEqual(result1, nodes[1],'選択回数が0のNodeが選択される')

    def test_select_low_value(self):
        nodes = list()
        nodes.append(MockNode(10, 0.0, 0.0))
        nodes.append(MockNode(10, 0.0, 0.0))
        nodes.append(MockNode(10, 0.0, 0.0))
        nodes.append(MockNode(10, -0.1, 0.0))
        selector = UCB1NextNodeSelector()
        result1 = selector.select(nodes)
        self.assertEqual(result1, nodes[3],'価値が低いNodeが選択される')

    def test_select_fewer_selected(self):
        nodes = list()
        nodes.append(MockNode(20, 0.1, 0.0))
        nodes.append(MockNode(20, 0.1, 0.0))
        nodes.append(MockNode(19, 0.1, 0.0))
        nodes.append(MockNode(20, 0.1, 0.0))
        selector = UCB1NextNodeSelector()
        result1 = selector.select(nodes)
        self.assertEqual(result1, nodes[2],'同じ価値なら選択回数が少ないNodeが選択される')

class Test_PolicyUCTNextNodeSelector(unittest.TestCase):
    def test_select_higher_policy_node(self):
        nodes = list()
        nodes.append(MockNode(0, 0.0, 0.0))
        nodes.append(MockNode(0, 0.0, 0.0))
        nodes.append(MockNode(0, 0.0, 0.1))
        nodes.append(MockNode(0, 0.0, 0.0))
        selector = PolicyUCTNextNodeSelector()
        result1 = selector.select(nodes)
        self.assertEqual(result1, nodes[2],'選択回数がすべて0の場合はポリシー値が高いNodeが選択される')

    def test_select_same_value_node(self):
        nodes = list()
        nodes.append(MockNode(1, 0.0, 0.0))
        nodes.append(MockNode(1, 0.0, 0.0))
        nodes.append(MockNode(1, 0.0, 0.0))
        nodes.append(MockNode(1, 0.0, 0.0))

        selector = PolicyUCTNextNodeSelector()
        count = 4000
        def get_select_index(selected_node):
            for i, node in enumerate(nodes):
                if node == selected_node:
                    return i
            return -1
        selected_counts = np.zeros(shape=(4,), dtype=np.int32)
        for _ in range(count):
            result1 = selector.select(nodes)
            selected_counts[get_select_index(result1)] += 1
        ratios = selected_counts / count
        self.assertAlmostEqual(ratios[0], 1.0 / 4, delta=0.1)
        self.assertAlmostEqual(ratios[1], 1.0 / 4, delta=0.1)
        self.assertAlmostEqual(ratios[2], 1.0 / 4, delta=0.1)
        self.assertAlmostEqual(ratios[3], 1.0 / 4, delta=0.1)

    def test_select_low_value_node(self):
        nodes = list()
        nodes.append(MockNode(1, 0.0, 0.0))
        nodes.append(MockNode(1, -0.1, 0.0))
        nodes.append(MockNode(1, 0.0, 0.0))
        nodes.append(MockNode(1, 0.0, 0.0))
        selector = PolicyUCTNextNodeSelector()
        result1 = selector.select(nodes)
        self.assertEqual(result1, nodes[1],'価値が低いNodeが選択される')

    def test_select_search_value_node(self):
        nodes = list()
        nodes.append(MockNode(1, 0.0, 0.1))
        nodes.append(MockNode(1, 0.0, 0.1))
        nodes.append(MockNode(1, 0.0, 0.1))
        nodes.append(MockNode(1, -0.1, 0.1))
        selector = PolicyUCTNextNodeSelector()
        for i in range(40):
            selected_node = selector.select(nodes)
            selected_node.selected_count += 1
        print('{},{},{},{}'.format(nodes[0].selected_count, nodes[1].selected_count, nodes[2].selected_count, nodes[3].selected_count))
        self.assertGreater(nodes[3].selected_count, nodes[0].selected_count)
        self.assertGreater(nodes[3].selected_count, nodes[1].selected_count)
        self.assertGreater(nodes[3].selected_count, nodes[2].selected_count)

    def test_dirichlet_node_selector(self):
        nodes = list()
        nodes.append(MockNode(1, 0.0, 1.0/4))
        nodes.append(MockNode(1, 0.0, 1.0/4))
        nodes.append(MockNode(1, 0.0, 1.0/4))
        nodes.append(MockNode(1, 0.0, 1.0/4))
        selector=WithDirichletPolicyUCTNextNodeSelector()
        count = 20
        for _ in range(count):
            selected_node = selector.select(nodes)
            selected_node.selected_count += 1

        print('{},{},{},{}'.format(nodes[0].selected_count, nodes[1].selected_count, nodes[2].selected_count, nodes[3].selected_count))
        nodes2 = list()
        nodes2.append(MockNode(1, 0.0, 1.0/4))
        nodes2.append(MockNode(1, 0.0, 1.0/4))
        nodes2.append(MockNode(1, 0.0, 1.0/4))
        nodes2.append(MockNode(1, 0.0, 1.0/4))
        for _ in range(count):
            selected_node = selector.select(nodes2)
            selected_node.selected_count += 1
        print('{},{},{},{}'.format(nodes2[0].selected_count, nodes2[1].selected_count, nodes2[2].selected_count, nodes2[3].selected_count))

        self.assertFalse(
            nodes[0].selected_count == nodes2[0].selected_count and
            nodes[1].selected_count == nodes2[1].selected_count and
            nodes[2].selected_count == nodes2[2].selected_count and
            nodes[3].selected_count == nodes2[3].selected_count)


class TestGameBoard(GameBoard):
    def __init__(self, 
                turn: int, 
                last_action : int, 
                activation_space: int = 4, 
                result: GameRelativeResult = GameRelativeResult.draw,
                selecteds: list = None,
                history: list = list()):
        super().__init__(turn, last_action)
        self.activation_space = activation_space
        self.result = result
        if selecteds is None:
            self.selecteds = np.array(range(0, activation_space), dtype=np.int32)
        else:
            self.selecteds = selecteds
        self.history = history
    # 行動空間サイズ
    def get_output_size(self)->int:
        return self.activation_space
    # historyに記憶させる形状に変換
    def to_hisotry_record(self)->any:
        pass # ignore

    # モデルに渡す形状に変換
    def reshape_to_input(self)->np.ndarray:
        pass # ignore

    # historyからモデルに渡す形状に変換
    def reshape_history_to_input(self, history) -> any:
        pass # ignore

    # 次の状態への遷移
    def transit_next(self, action)-> Tuple['TestGameBoard', bool]:
        new_selecteds = np.array([], dtype=np.int32)
        history = self.history.copy()
        history.append(action)
        for v in self.selecteds:
            if v == action:
                continue
            new_selecteds = np.append(new_selecteds, v)
        return TestGameBoard(
            turn=self.turn + 1, 
            last_action=action, 
            activation_space=self.activation_space,
            result=self.result,
            selecteds=new_selecteds,
            history=history), True
    
    def to_state_key(self)->str:
        return '{0}'.format(self.history)

    def get_legal_actions(self)->np.ndarray:
        return self.selecteds

    def judge_last_action(self)-> Tuple[bool, GameRelativeResult]:
        if self.activation_space == self.turn:
            return True, self.result
        return False, None

class TestPredictor(Predictor):
    def __init__(self, value: float=0.2,ts_dict=ThreadSafeDict()):
        super().__init__(ts_dict=ts_dict)
        self.value = value
    def predict_core(self, game_board: GameBoard) -> Prediction:
        policies = np.zeros(shape=(game_board.get_output_size(),), dtype=np.float32)
        legal_actions=game_board.get_legal_actions()
        p = 1.0 / len(legal_actions)
        for action in legal_actions:
            policies[action] = p
        return DualNetworkPrediction(policies, self.value)

class TestMctsNode(unittest.TestCase):
    def test_random_mcts_node(self):
        action_space = 4
        game_board = TestGameBoard(0, -1, action_space, GameRelativeResult.draw)
        test_node = RandomMctsNode(game_board=game_board, is_root=True, expand_limit=5, node_selector=UCB1NextNodeSelector())
        count = 100
        total_count = count * action_space
        ratio = 0.1
        low_limit = int(float(count) * (1.0 - ratio))
        high_limit = int(float(count) * (1.0 + ratio))

        for i in range(0, total_count):
            test_node.evaluate()
        self.assertEqual(total_count, test_node.get_selected_count())
        self.assertEqual(action_space, len(test_node.child_nodes))

        for child_node in test_node.child_nodes:
            self.assertTrue(child_node.get_selected_count() > low_limit)
            self.assertTrue(child_node.get_selected_count() < high_limit)



    def test_pvnetwork_mcts_node_flat(self):

        action_space = 4
        game_board = TestGameBoard(0, -1, action_space, GameRelativeResult.draw)
        value_base = 0.2
        predictor = TestPredictor(value=value_base)

        test_node = PolicyValueNetworkMctsNode(
            game_board=game_board, 
            is_root=True, 
            predictor_alpha=predictor,
            predictor_beta=predictor,
            policy=0.01,
            node_selector=PolicyUCTNextNodeSelector(),
            child_node_selector=PolicyUCTNextNodeSelector())
        
        # ルート展開直後
        test_node.evaluate()
        self.assertEqual(action_space, len(test_node.child_nodes))
        self.assertAlmostEqual(test_node.get_value(), value_base)
        for child_node in test_node.child_nodes:
            self.assertAlmostEqual(child_node.get_policy(), 1.0 / action_space)
            self.assertAlmostEqual(child_node.get_value(), 0.0)

        # 各ノードのポリシーが同じなので、各一回ずつ評価（展開）
        for i in range(0, action_space):
            test_node.evaluate()
        self.assertAlmostEqual(test_node.get_value(), value_base - (value_base * action_space))
        for child_node in test_node.child_nodes:
            self.assertAlmostEqual(child_node.get_policy(), 1.0 / action_space)
            self.assertAlmostEqual(child_node.get_selected_count(), 1)
            self.assertAlmostEqual(child_node.get_value(), value_base)

        test_count = 10000
        for i in range(0, action_space * test_count):
            test_node.evaluate()

        ratios = [x.get_selected_count() / (action_space * test_count) for x in test_node.child_nodes]
        for ratio in ratios:
            self.assertAlmostEqual(ratio, 1.0 / action_space, delta=0.1)


    def test_pvnetwork_mcts_node_value(self):

        action_space = 4
        game_board = TestGameBoard(0, -1, action_space, GameRelativeResult.draw)
        value_base = 0.2
        ts_dict = ThreadSafeDict()
        # 特定の局面のみ評価値を変更
        ts_dict['[3]'] = DualNetworkPrediction(np.array([1.0/3, 1.0/3, 1.0/3, 0.0], dtype=np.float32), -0.5)
        predictor = TestPredictor(value=value_base, ts_dict=ts_dict)

        test_node = PolicyValueNetworkMctsNode(
            game_board=game_board, 
            is_root=True, 
            predictor_alpha=predictor,
            predictor_beta=predictor,
            policy=0.01,
            node_selector=PolicyUCTNextNodeSelector(),
            child_node_selector=PolicyUCTNextNodeSelector())
        
        # ルート展開直後
        test_node.evaluate()
        self.assertEqual(action_space, len(test_node.child_nodes))
        self.assertAlmostEqual(test_node.get_value(), value_base)
        for child_node in test_node.child_nodes:
            self.assertAlmostEqual(child_node.get_policy(), 1.0 / action_space)
            self.assertAlmostEqual(child_node.get_value(), 0.0)

        test_count = 40
        for i in range(0, action_space * test_count):
            test_node.evaluate()

        ratios = [x.get_selected_count() / (action_space * test_count) for x in test_node.child_nodes]
        for i, ratio in enumerate(ratios):
            print('i={}, ratio={}'.format(i, ratio))
            if i == 3:
                self.assertGreater(ratio, 1.0 / action_space)
            else:
                self.assertLess(ratio, 1.0 / action_space)

    def test_pvnetwork_mcts_node_policy(self):

        action_space = 4
        game_board = TestGameBoard(0, -1, action_space, GameRelativeResult.draw)
        value_base = 0.2
        ts_dict = ThreadSafeDict()
        # 特定の局面のみ評価値を変更
        ts_dict['[]'] = DualNetworkPrediction(np.array([1.0/5, 2.0/5, 1.0/5, 1.0/5, 0.0], dtype=np.float32), value_base)
        predictor = TestPredictor(value=value_base, ts_dict=ts_dict)

        test_node = PolicyValueNetworkMctsNode(
            game_board=game_board, 
            is_root=True, 
            predictor_alpha=predictor,
            predictor_beta=predictor,
            policy=0.01,
            node_selector=PolicyUCTNextNodeSelector(),
            child_node_selector=PolicyUCTNextNodeSelector())
        
        # ルート展開直後
        test_node.evaluate()
        self.assertEqual(action_space, len(test_node.child_nodes))
        self.assertAlmostEqual(test_node.get_value(), value_base)
        for i, child_node in enumerate(test_node.child_nodes):
            if i == 1:
                self.assertGreater(child_node.get_policy(), 2.0 / 5)
            else:
                self.assertAlmostEqual(child_node.get_policy(), 1.0 / 5)
            self.assertAlmostEqual(child_node.get_value(), 0.0)

        test_count = 40
        for i in range(0, action_space * test_count):
            test_node.evaluate()

        ratios = [x.get_selected_count() / (action_space * test_count) for x in test_node.child_nodes]
        for i, ratio in enumerate(ratios):
            print('i={}, ratio={}'.format(i, ratio))
            if i == 1:
                self.assertGreater(ratio, 1.0 / action_space)
            else:
                self.assertLess(ratio, 1.0 / action_space)




'''
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

'''