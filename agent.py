


class Agent:
    def __init__(self, brain):
        self.brain = brain
    def select_action(self, board)->int:
        return self.brain.select_action(board)
    def get_brain_name(self)->str:
        return self.brain.get_name()
    