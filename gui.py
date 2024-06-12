import tkinter as tk
from game_board import GameBoard
from game import GameEnv
from agent import Agent
from mini_max import AlphaBetaBrain
from network_common import NetworkBrain

import concurrent.futures

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
        self.executor = concurrent.futures.ThreadPoolExecutor(1)
        self.future = None
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
        if self.game_board.is_done():
            return
        if self.game_board.is_first_player_turn():
            if type(self.first_agent.brain) is not HumanGuiBrain:
                if self.future is None:
                    self.future = self.executor.submit(lambda : self.act_network(self.first_agent))
                    self.polling()
        else:
            if type(self.second_agent.brain) is not HumanGuiBrain:
                if self.future is None:
                    self.future = self.executor.submit(lambda : self.act_network(self.second_agent))
                    self.polling()

    def act_network(self, agent : Agent)->GameBoard:
        print("begin select action")
        selected = agent.select_action(self.game_board)
        if type(agent.brain) is NetworkBrain:
            policies = agent.brain.get_last_policies()
            print("selected:{} ,policiy:{}".format(selected, policies))
        else:
            print("selected:{}".format(selected))
        next_board,succeed = self.game_board.transit_next(selected)
        if not succeed:
            return
        return next_board
    
    def polling(self):
        if self.future is None:
            return
        if self.future.done():
            next_board = self.future.result()
            self.future = None
            self.update(next_board)
            return
        self.master.after(200, self.polling)


    def update(self, next_board : GameBoard):
        self.game_board = next_board
        self.on_draw()
        done, result = self.game_board.judge_last_action()
        if done:
            print(result)
        self.try_network_action()

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
        next_board, succeed = self.game_board.transit_next(action)
        if not succeed:
            return
        self.game_board = next_board
        self.on_draw()
        done, result = self.game_board.judge_last_action()
        if done:
            print(result)
            return

        self.try_network_action()


def run_gui(board : GameBoard, first_agent : Agent, second_agent : Agent):
    root = tk.Tk()
    app = Application(board, first_agent, second_agent, root)
    app.mainloop()

