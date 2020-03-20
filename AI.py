import os
from abc import ABC, abstractmethod
from random import random

import numpy as np
import tensorflow.keras.losses as l
from tensorflow import function
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


class AI(ABC):
    def __init__(self, speed_limit, widget, ball):
        # speed limit is given in height/sec
        self.speed_limit = speed_limit
        self.widget = widget
        self.ball = ball

    @abstractmethod
    def play(self, dt):
        pass

    @abstractmethod
    def notify_end(self, won):
        pass

    @abstractmethod
    def on_pong(self):
        pass


class Heuristic(AI):
    def notify_end(self, won):
        pass

    def __init__(self, speed_limit, widget, ball):
        super().__init__(speed_limit, widget, ball)
        self.rand = random()

    def play(self, dt):
        self.widget.center_y += self._decide(dt, self.ball.velocity,
                                             self.ball.pos, self.widget.size,
                                             self.widget.center_y)

    def on_pong(self):
        # print("Pong: Heuristic")
        self.rand = random()

    def _decide(self, dt, ball_vel, ball_pos, paddle_size, center_y):
        desired = ball_pos[1] - center_y
        error_margin = (ball_vel[1] / abs(ball_vel[1] + 1e-6)) * paddle_size[1] * 0.1 * self.rand * 3
        desired += error_margin
        if self.speed_limit == -1 or abs(desired / dt) <= self.speed_limit:
            return desired
        else:
            return (self.speed_limit * desired / abs(desired)) * dt


class NeuralNet(AI):
    def on_pong(self):
        # print("Pong: NN")
        self.memory_lbls += [1.0] * (len(self.memory_feats) - len(self.memory_lbls))

    def __init__(self, speed_limit, my_widget, enemy_widget, ball, game_size, save_path=".\\saved_models\\model",
                 training=True):
        super().__init__(speed_limit, my_widget, ball)
        self.save_path = save_path
        self.game_size = game_size
        self.enemy_widget = enemy_widget
        self.training = training
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.memory_feats = []
        self.memory_lbls = []
        self.memory_pred = []
        K.set_learning_phase(0)
        self.random = 0.1
        self.random_decay = 0.99

    def random_next(self):
        self.random *= self.random_decay

    def _build_model(self):
        """
        features:
            delta_t: absolute
            ball_x: relative to x_max
            ball_y: relative to y_max
            ball_vel_x: absolute
            ball_vel_y: absolute
            own_x: relative to x_max
            own_y: relative to y_max
            enemy_x: relative to x_max
            enemy_y: relative to y_max
            widget_size_y
        outputs:
            100 Q-values, first for being in the lower 1% of the screen, 100th for being in the top 1% of the screen
        """
        inps = Input(shape=(10,))
        out = Dense(5, activation="elu")(inps)
        out = Dense(256, activation="elu")(out)
        out = Dense(128, activation="elu")(out)
        out = Dense(50, activation="linear")(out)
        model = Model(inputs=inps, outputs=out)
        if not os.path.isdir(self.save_path):
            model.save_weights(self.save_path)

        else:
            l.rl_loss = self.rl_loss
            model.load_weights(self.save_path)

        if self.training:
            model.compile(loss="MSE", optimizer=SGD(learning_rate=0.001))

        return model

    def play(self, dt):

        x_max = float(self.game_size[0])
        y_max = float(self.game_size[1])

        ball_x = self.ball.pos[0] / x_max
        ball_y = self.ball.pos[1] / y_max
        ball_vel_x = self.ball.velocity[0]
        ball_vel_y = self.ball.velocity[1]
        own_x = self.widget.center_x / x_max
        own_y = self.widget.center_y / y_max
        enemy_x = self.enemy_widget.center_x / x_max
        enemy_y = self.enemy_widget.center_y / y_max
        widget_size_y = self.widget.size[1]
        self.widget.center_y += self._decide(dt, ball_x, ball_y, ball_vel_x, ball_vel_y, own_x, own_y, enemy_x, enemy_y,
                         widget_size_y) * self.speed_limit
        if self.widget.center_y < 0:
            self.widget.center_y = 0
        elif self.widget.center_y > self.game_size[1]:
            self.widget.center_y = self.game_size[1]

    def _decide(self, dt, ball_x, ball_y, ball_vel_x, ball_vel_y, own_x, own_y, enemy_x, enemy_y, widget_size_y):
        """
        :return: new relative y between 0 (top) and 1 (bottom)
        """
        X = np.array([[dt, ball_x, ball_y, ball_vel_x, ball_vel_y, own_x, own_y, enemy_x, enemy_y, widget_size_y]],
                     dtype=np.float32)
        out = np.squeeze(self.model(X))
        self.memory_feats.append(X)
        self.memory_pred.append(out)
        if np.random.random() < self.random:
            out = np.random.choice(np.arange(0, 1, 0.2))
        return out

    def notify_end(self, won):
        self.memory_lbls += [2 * float(won) - 1] * (len(self.memory_feats) - len(self.memory_lbls))

    def train(self):
        # print("training started!")
        K.set_learning_phase(1)
        X = np.array(np.squeeze(self.memory_feats))
        y = np.array(self.memory_lbls)
        self.model.fit(X, y, epochs=1, verbose=1, batch_size=len(X))
        self.model.save_weights(self.save_path)

    @staticmethod
    @function
    def rl_loss(y_true, y_pred):
        return K.sum(y_true * y_pred)
