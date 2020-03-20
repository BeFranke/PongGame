from abc import ABC


class Player(ABC):
    pass


class AI(ABC, Player):
    pass


class HeuristicAI(AI):
    pass


class NeuralNet(AI):
    pass


class Human(Player):
    pass
