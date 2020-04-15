from abc import ABC, abstractmethod
import numpy as np


class Player(ABC):
    def __init__(self, id, **kwargs):
        self.id = id

    @abstractmethod
    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray) -> float:
        pass


class Heuristic(Player):
    def __init__(self, id, **kwargs):
        super().__init__(id, **kwargs)

    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray) -> float:
        return ball_pos[1]



class NeuralNet(Player):
    def __init__(self, id, **kwargs):
        super().__init__(id, **kwargs)


class Human(Player):
    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray) -> float:
        # only here to make interaction easier
        return player1_pos[1] if self.id == 0 else player2_pos[1]

    def __init__(self, id, **kwargs):
        super().__init__(id, **kwargs)
