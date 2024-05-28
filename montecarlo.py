import numpy as np
import math
from game_board import GameBoard, GameRelativeResult, GameResult
from brains import Brain

def playout(game_board: GameBoard):
    # 相手が行動した直後の状態を受け取る。相手が勝ってたら自分の負け。
    done, result = game_board.judge_last_action()
    if done:
        if result == GameRelativeResult.draw:
            return 0
        elif result == GameRelativeResult.win_last_play_player:
            return -1
        else:
            return 1
    actions = game_board.get_legal_actions()
    next_state, _ = game_board.transit_next(np.random.choice(actions))
    return -playout(next_state)

class PrimeMonteCarloBrain:
    def __init__(self, count = 10):
        self.count = count
        self.name = "PrimeMonteCarloBrain"
    def get_name(self):
        return self.name
    def select_action(self, board):
        actions = board.get_legal_actions()
        values = np.zeros(shape=(actions.shape[0],), dtype=np.int16)
        for i, action in enumerate(actions):
            for _ in range(self.count):
                next_state, _ = board.transit_next(action)
                values[i] += -playout(next_state)
        selected = actions[np.argmax(values)]
        #s = ''
        #for i, action in enumerate(actions):
        #    s += str(action)
        #    s += ":"
        #    s += str(values[i])
        #    s += " "
        #s += " ...selected:" + str(selected)
        #print(s)
        return selected



def monte_carlo_action(board, count = 100):
    class Node:
        def __init__(self, board, w = 0, n = 0, expand_limit = 10):
            self.board = board
            self.w = w
            self.n = n
            self.child_nodes = None
            self.expand_limit = expand_limit
        def output(self):
            print('>> ====node info====')
            self.board.output()
            print('w={0},n={1}'.format(self.w, self.n))
            print('<< ====node info====')
        def evaluate(self):
            # 相手が行動した直後の状態を受け取る。相手が勝ってたら自分の負け。
            done, result = board.judge_last_action()
            if done:
                value = 0
                if result == GameRelativeResult.draw:
                    pass
                elif result == GameRelativeResult.win_last_play_player:
                    value -= 1
                else:
                    value += 1
                self.w += value
                self.n += 1
                return value
            if not self.child_nodes:
                value = playout(self.board)
                self.w += value
                self.n += 1

                if self.n == self.expand_limit:
                    self.expand()
                return value
            else:
                value = -self.next_child_node().evaluate()
                self.w += value
                self.n += 1
                return value
        def expand(self):
            actions = self.board.get_legal_actions()
            self.child_nodes = []
            for action in actions:
                next_board, succeed = self.board.transit_next(action)
                if succeed:
                    self.child_nodes.append(Node(next_board, 0, 0, self.expand_limit))
        def next_child_node(self):
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = np.zeros(shape=(0,))
            for child_node in self.child_nodes:
                value = -child_node.w / child_node.n + (2*math.log(t)/ child_node.n)**0.5
                ucb1_values = np.append(ucb1_values, value)

            selected_node_index = np.argmax(ucb1_values)
            #print(">>>> enter node selection")
            #for i, cn in enumerate(self.child_nodes):
            #    print("--- " + str(i))
            #    print("node value:" + str(ucb1_values[i]))
            #    cn.output()
            #print("<<<< enter node selection:"  + str(selected_node_index))
            return self.child_nodes[selected_node_index]

    root_node = Node(board)
    root_node.expand()

    for _ in range(count):
        root_node.evaluate()
    
    actions = board.get_legal_actions()
    n_list = np.zeros(shape=(0,), dtype=np.int16)
    for c in root_node.child_nodes:
        n_list = np.append(n_list, c.n)
    selected = actions[np.argmax(n_list)]
    #s = ''
    #for i, action in enumerate(actions):
    #    s += str(action)
    #    s += ":"
    #    s += str(n_list[i])
    #    s += " "
    #s += " ...selected:" + str(selected)
    #print(s)
    return selected


class MonteCarloBrain(Brain):
    def __init__(self, count = 10):
        super().__init__()
        self.count = count
        self.name = "MonteCarloBrain"
    def get_name(self):
        return self.name
    def select_action(self, board):
        selected = monte_carlo_action(board)
        return selected

