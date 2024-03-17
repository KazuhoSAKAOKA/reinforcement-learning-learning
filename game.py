from game_board import GameBoard, GameRelativeResult, GameResult
from agent import Agent
from typing import Callable

def do_empty1(board : GameBoard):
    pass
def do_empty2(board : GameBoard, action : int):
    pass
def do_empty3(i: int, board : GameBoard, result : GameResult):
    pass

def output_progress(board : GameBoard):
    print(board)
def selected_progress(board : GameBoard, action : int):
    print("Selected action: {0}".format(action))
    print(board)
def episode_progress(i: int, board : GameBoard, result : GameResult):
    print("{0} Game result: {1}".format(i + 1, result))
    print(board)

class GameEnv:
    def __init__(self, board : GameBoard, first_agent : Agent, second_agent : Agent , prev_action_callback :Callable[[GameBoard], None] = do_empty1, selected_action_callback : Callable[[GameBoard, int],None]= do_empty2, episode_callback : Callable[[int, GameBoard, GameResult], None] = do_empty3):
        self.board = board
        self.board.reset()
        self.first_agent = first_agent
        self.second_agent = second_agent
        self.prev_action_callback = prev_action_callback
        self.selected_action_callback = selected_action_callback
        self.episode_callback = episode_callback
        self.index = 0
    def reset(self):
        self.index = 0
        self.board.reset()

    
    def play(self) -> GameResult:
        self.board.reset()
        current = self.board
        while True:
            done, result = current.judge_last_action()
            if done:
                break

            self.prev_action_callback(current)

            if current.get_turn() % 2 == 0:
                action = self.first_agent.select_action(current)
            else:
                action = self.second_agent.select_action(current)
            succeed, next_board = current.transit_next(action)
            if not succeed:
                raise Exception("Invalid action")

            current = next_board
            self.selected_action_callback(current, action)
        absolute_result = current.convert_to_result(result)
        self.episode_callback(self.index, current, absolute_result)
        return absolute_result
    
    def play_n(self, n: int) -> (int, int, int):
        self.index = 0
        first_player_win = 0
        second_player_win = 0
        draw = 0
        for _ in range(n):
            result = self.play()
            if result == GameResult.win_first_player:
                first_player_win += 1
            elif result == GameResult.win_second_player:
                second_player_win += 1
            else:
                draw += 1
            self.index += 1
        print("First player {0} win: {1}, Second player {2} win: {3}, Draw: {4}".format(self.first_agent.get_brain_name(), first_player_win, self.second_agent.get_brain_name(), second_player_win, draw))
        return first_player_win, second_player_win, draw




if __name__=='__main__':
    pass   