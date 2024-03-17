


class Agent:
    def __init__(self, brain):
        self.brain = brain
    def select_action(self, board):
        return self.brain.select_action(board)
    def get_brain_name(self):
        return self.brain.get_name()
    