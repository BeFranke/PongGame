from typing import Union, List

import numpy as np
from scipy.spatial.transform import Rotation as R

from AI import *
from settings import Config


class PongGame:
    SCORE_TO_WIN = Config.get('points_to_win')

    def __init__(self, size=(1920, 1080)):
        player = Config.get('players')
        self.player_list: List[Union[AI, None]] = [None] * 2
        self.ball = PongBall()
        widgets = [MockWidget(0, size=size), MockWidget(1, size=size)]
        self.height = size[1]
        self.width = size[0]
        self.size = size
        self.player1_score = 0
        self.player2_score = 0
        for i in range(2):
            if isinstance(player[i], list):
                if player[i][0] == "Heuristic":
                    self.player_list[i] = Heuristic(player[i][1], widgets[0], self.ball)
                elif player[i][0] == "NeuralNet":
                    self.player_list[i] = NeuralNet(player[i][1], widgets[1], widgets[0], self.ball, self.size)
                else:
                    raise Exception("Config Error: Malformatted AI entry!")
            else:
                raise Exception("Config Error: AI Type not understood!")

    def update(self, dt):
        for player in self.player_list:
            player.play(dt)

        self.ball.move()

        if (self.ball.y < 0) or (self.ball.y > self.height):
            self.ball.velocity[1] *= -1

        elif self.ball.x < 0:
            self.player2_score += 1
            self.player_list[0].notify_end(False)
            self.player_list[1].notify_end(True)
            print("Player 2 scored!")

            if self.player2_score == self.SCORE_TO_WIN:
                return 2

            self.ball = PongBall()

        elif self.ball.x > self.width:
            self.player1_score += 1
            self.player_list[0].notify_end(True)
            self.player_list[1].notify_end(False)
            print("Player 1 scored!")

            if self.player1_score == self.SCORE_TO_WIN:
                return 1

            self.ball = PongBall()

        return 0


class PongBall:
    def __init__(self, size=(1920, 1080)):
        self.y = size[1] * 0.5
        self.x = size[0] * 0.5
        a = np.random.rand() * np.pi * 2
        self.velocity = np.array([5, 0]) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def move(self):
        self.x += self.velocity[0]
        self.y += self.velocity[1]

    @property
    def pos(self):
        return np.array([self. x, self.y])

    @pos.setter
    def pos(self, p):
        self.x = p[0]
        self.y = p[1]


class MockWidget:
    def __init__(self, player, size=(1920, 1080)):
        self.center_x = size[0] * 1/10 if player == 0 else size[0] * 9/10
        self.center_y = size[1] * 0.5
        self.size = [25, 200]


if __name__ == "__main__":
    training_data = []
    training_labels = []
    for epoch in range(10):
        print(f"starting data generation {epoch} of 50")
        Config.cfgf = "training.cfg"
        Config.load()
        game = PongGame()
        # game.player_list[1].memory_feats += training_data
        # game.player_list[1].memory_lbls += training_labels

        while game.update(1/300) == 0:
            pass
        print("started training!")
        game.player_list[1].train()
        # training_data += game.player_list[1].memory_feats
        # training_labels += game.player_list[1].memory_lbls
