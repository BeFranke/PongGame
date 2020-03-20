import numpy as np


class GameModel:
    def __init__(self):
        self.player_1_pos = np.array([0.1, 0.5])
        self.player_2_pos = np.array([0.9, 0.5])
        self.ball_pos = np.array([0.5, 0.5])
        self.ball_vel = np.array([1, 0])

    def get_initial_positions(self):
        return self.player_1_pos, self.player_2_pos, self.ball_pos
