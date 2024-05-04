from brains import Brain


class Agent:
    def __init__(self, brain : Brain, name : str = 'no name'):
        self.brain = brain
        self.name = name
    def select_action(self, board)->int:
        return self.brain.select_action(board)
    def get_name(self)->str:
        return 'agent :{0}, brain:{1}'.format(self.name, self.brain.get_name())
    def __repr__(self) -> str:
        return self.get_name()
    