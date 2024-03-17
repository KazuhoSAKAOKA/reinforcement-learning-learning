from game_board import GameRelativeResult

def evaluate_board_to_final(board, depth):
    '''
    ゲームの終了までの状態を評価する
    '''
    # 相手が行動した直後の状態を受け取る。相手が勝ってたら自分の負け。
    done, result = board.judge_last_action()
    if done:
        if result == GameRelativeResult.draw:
            return True, 0
        elif result == GameRelativeResult.win_last_play_player:
            return True, -1
        else:
            return True, 1
    return False, 0

def mini_max(board, depth, evaluate_board):
    evaluated, value = evaluate_board(board, depth)
    if evaluated:
        return value
    
    best_score = -float('inf')
    for action in board.get_legal_actions():
        succeed, next_state = board.transit_next(action)
        if succeed:
            score = -mini_max(next_state, depth + 1, evaluate_board)
            if score > best_score:
                best_score = score
    return best_score


def alpha_beta(board, alpha, beta, depth, evaluate_board_alpha, evaluate_board_beta):
    evaluated, value = evaluate_board_alpha(board, depth)
    if evaluated:
        return value
    for action in board.get_legal_actions():
        succeed, next_state = board.transit_next(action)
        if succeed:
            score = -alpha_beta(next_state, -beta, -alpha, depth + 1, evaluate_board_beta, evaluate_board_alpha)
            if score > alpha:
                alpha = score
            if alpha >= beta:
                return alpha
    return alpha



class MiniMaxBrain:
    def __init__(self):
        pass
        self.name = "MiniMax"
    def get_name(self):
        return self.name
    def select_action(self, board):
        best_action = 0
        best_score = -float('inf')
        str = ['','']
        for action in board.get_legal_actions():
            succeed, next_state = board.transit_next(action)
            if succeed:
                score = -mini_max(next_state)
                if score > best_score:
                    best_action = action
                    best_score = score
        #        str[0] = '{}{:2d},'.format(str[0], action)
        #        str[1] = '{}{:2d},'.format(str[1], score)
        #print('action:', str[0], '\nscore: ', str[1], '\n')
        return best_action

class AlphaBetaBrain:
    def __init__(self):
        pass
        self.name = "AlphaBetaBrain"
    def get_name(self):
        return self.name
    def select_action(self, board):
        best_action = 0
        alpha = -float('inf')
        #str = ['','']
        for action in board.get_legal_actions():
            succeed, next_state = board.transit_next(action)
            if succeed:
                score = -alpha_beta(next_state, -float('inf'), -alpha, 0, evaluate_board_to_final, evaluate_board_to_final)
                if score > alpha:
                    best_action = action
                    alpha = score
        #        str[0] = '{}{:2d},'.format(str[0], action)
        #        str[1] = '{}{:2d},'.format(str[1], score)
        #print('action:', str[0], '\nscore: ', str[1], '\n')
        return best_action

