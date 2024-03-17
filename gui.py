import tkinter as tk
from game_board import GameBoard
from game import GameEnv
from agent import Agent
from mini_max import AlphaBetaBrain
from network_common import NetworkBrain, predict, load_model
from tictactoe_board import TicTacToeBoard
from tictactoe_network import MODEL_FILE_BEST
import threading

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
        self.canvas = tk.Canvas(self.master, width=400, height=400)
        self.canvas.bind("<Button-1>", self.click)
        self.pack()
        self.first_agent = first_agent
        self.second_agent = second_agent
        self.policies = None
        self.thread = None


        self.on_draw()
        self.try_network_action()

    def on_draw(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 400, 400, fill="white")
        #for x in range(self.game_board.board_size):
        #    self.canvas.create_line(0, x * 100, 300, x * 100, fill="black")
        #    self.canvas.create_line(x * 100, 0, x * 100, 300, fill="black")

        for y in range(self.game_board.board_size):
            for x in range(self.game_board.board_size):
                if self.policies is not None:
                    index = y * self.game_board.board_size + x
                    color = int(255 * self.policies[index])
                    self.canvas.create_rectangle(x * 100, y * 100, (x + 1) * 100, (y + 1) * 100, outline="black", fill="#%02x%02x%02x" % (color, 255, 255))
                value =self.game_board.get_cell(y, x)
                if value == 1:
                    self.canvas.create_oval(x * 100, y * 100, (x + 1) * 100, (y + 1) * 100, fill="black", background="transparent")
                if value == 2:
                    self.canvas.create_oval(x * 100, y * 100, (x + 1) * 100, (y + 1) * 100, fill="white", background="transparent")
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
        selected = agent.select_action(self.game_board)
        self.policies = agent.brain.get_last_policies()
        succeed, next_board = self.game_board.transit_next(selected)
        if not succeed:
            return
        self.game_board = next_board
        self.on_draw()
    def click(self, event):
        if not self.is_player_turn():
            return
        x = event.x // 100
        y = event.y // 100
        action = y * self.game_board.board_size + x
        actives = self.game_board.get_legal_actions()
        if action not in actives:
            return

        succeed, next_board = self.game_board.transit_next(action)
        if not succeed:
            return
        self.game_board = next_board
        self.on_draw()
        
        self.try_network_action()


if __name__ == "__main__":
    root = tk.Tk()
    board = TicTacToeBoard()
    model = load_model(MODEL_FILE_BEST)
    first_agent = Agent(NetworkBrain(0.1, 10, lambda x: predict(model, x), lambda x: predict(model, x)))
    second_agent = Agent(HumanGuiBrain())
    app = Application(board, first_agent, second_agent, master=root)
    app.mainloop()