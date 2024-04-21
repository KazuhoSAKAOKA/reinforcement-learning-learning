import tkinter as tk
from game_board import GameBoard
from game import GameEnv
from agent import Agent
from mini_max import AlphaBetaBrain
from network_common import NetworkBrain, predict, load_model
from tictactoe_board import TicTacToeBoard
from tictactoe_network import MODEL_FILE_BEST, MODEL_FILE_BEST_FIRST, MODEL_FILE_BEST_SECOND
from gomoku_board import GomokuBoard
from parameter import PARAM

import threading
import numpy as np
class HumanGuiBrain:
    def __init__(self):
        pass
    def get_name(self)->str:
        return "HumanGuiBrain"
    def select_action(self, board):
        pass

class Application(tk.Frame):
    def __init__(self, game_board : GameBoard, first_agent : Agent, second_agent : Agent, master=None):
        super().__init__(master)
        self.master = master
        self.game_board = game_board
        self.master.title("game")
        self.cell_size = 1500 / game_board.board_size

        self.canvas = tk.Canvas(self.master, width=1500, height=1500)
        self.canvas.bind("<Button-1>", self.click)
        self.pack()
        self.first_agent = first_agent
        self.second_agent = second_agent
        self.thread = None


        self.on_draw()
        self.try_network_action()

    def on_draw(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 1500, 1500, fill="white")
        for x in range(self.game_board.board_size):
            self.canvas.create_line(0, x * self.cell_size, self.cell_size * self.game_board.board_size, x * self.cell_size, fill="black")
            self.canvas.create_line(x * self.cell_size, 0, x * self.cell_size, self.cell_size * self.game_board.board_size, fill="black")

        actives = self.game_board.get_legal_actions()
        policy_index = 0
        for y in range(self.game_board.board_size):
            for x in range(self.game_board.board_size):
                back_color = "gray"
                index = y * self.game_board.board_size + x

#                if self.policies is not None:
#                    color = int(255 * self.policies[index])
#                    self.canvas.create_rectangle(x * 100, y * 100, (x + 1) * 100, (y + 1) * 100, outline="black", fill=back_color)
                value =self.game_board.get_cell(x, y)
                if value == 1:
                    self.canvas.create_oval(x * self.cell_size, y * self.cell_size, (x + 1) * self.cell_size, (y + 1) * self.cell_size, fill="black",  outline=back_color)
                if value == 2:
                    self.canvas.create_oval(x * self.cell_size, y * self.cell_size, (x + 1) * self.cell_size, (y + 1) * self.cell_size, fill="white", outline=back_color)
        self.canvas.pack()
    def is_player_turn(self):
        if self.game_board.is_first_player_turn():
            return type(self.first_agent.brain) is HumanGuiBrain
        else:
            return type(self.second_agent.brain) is HumanGuiBrain

    def try_network_action(self):
        if self.game_board.is_first_player_turn():
            if type(self.first_agent.brain) is NetworkBrain:
                self.act_network(self.first_agent)
        else:
            if type(self.second_agent.brain) is NetworkBrain:
                self.act_network(self.second_agent)

    def act_network(self, agent : Agent):
        
        selected = agent.brain.select_action(self.game_board)
        policies = agent.brain.get_last_policies()
        print("policiy:{}".format(policies))
        succeed, next_board = self.game_board.transit_next(selected)
        if not succeed:
            return
        self.game_board = next_board
        self.on_draw()
        done, result = self.game_board.judge_last_action()
        if done:
            print(result)

    def click(self, event):
        if not self.is_player_turn():
            return
        if self.game_board.is_done():
            return
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        action = (int)(y * self.game_board.board_size + x)
        actives = self.game_board.get_legal_actions()
        if action not in actives:
            return
        print("user select action:{}".format(action))
        succeed, next_board = self.game_board.transit_next(action)
        if not succeed:
            return
        self.game_board = next_board
        self.on_draw()
        done, result = self.game_board.judge_last_action()
        if done:
            print(result)
            return

        self.try_network_action()

def test_tictactoe():
    PARAM.alpha = 0.0
    root = tk.Tk()
    board = TicTacToeBoard()
    model = load_model(MODEL_FILE_BEST)
    model = load_model('./model/tictactoe/20240413091938.keras')
    network_agent = Agent(NetworkBrain(0, 200, lambda x: predict(model, x), lambda x: predict(model, x)))
    
    #first_model = load_model('./model/tictactoe/first/20240412054736.keras')
    #second_model = load_model('./model/tictactoe/second/20240412114427.keras')
    #network_agent = Agent(NetworkBrain(0, 20, lambda x: predict(second_model, x), lambda x: predict(first_model, x)))
    #network_agent = Agent(NetworkBrain(0, 200, lambda x: predict(first_model, x), lambda x: predict(second_model, x)))
    
    human_agent = Agent(HumanGuiBrain())
    app = Application(board, human_agent, network_agent, master=root)
    #app = Application(board, network_agent, human_agent, master=root)
    app.mainloop()    

def test_gomoku():
    root = tk.Tk()
    board = GomokuBoard(11)

    model = load_model(MODEL_FILE_BEST)
    #first_model = load_model(MODEL_FILE_BEST_FIRST)
    #second_model = load_model(MODEL_FILE_BEST_SECOND)
    network_agent = Agent(NetworkBrain(0.1, 10, lambda x: predict(model, x), lambda x: predict(model, x)))
    human_agent = Agent(HumanGuiBrain())
    app = Application(board, human_agent, network_agent, master=root)
    app.mainloop()    

if __name__ == "__main__":
    test_tictactoe()
