import os
from abc import ABC, abstractmethod
from random import random

import numpy as np
import tensorflow.keras.losses as l
from tensorflow import function
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


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
        self.rand = random()

    def _decide(self, dt, ball_vel, ball_pos, paddle_size, center_y):
        desired = ball_pos[1] - center_y
        error_margin = (ball_vel[1] / abs(ball_vel[1])) * paddle_size[1] * 0.1 * self.rand * 3
        desired += error_margin
        if self.speed_limit == -1 or abs(desired / dt) <= self.speed_limit:
            return desired
        else:
            return (self.speed_limit * desired / abs(desired)) * dt


class NeuralNet(AI):
    def on_pong(self):
        self.memory_lbls += [0.0] * (len(self.memory_feats) - len(self.memory_lbls))

    def __init__(self, speed_limit, my_widget, enemy_widget, ball, game_size, save_path=".\\saved_models\\model",
                 training=True):
        super().__init__(speed_limit, my_widget, ball)
        self.save_path = save_path
        self.game_size = game_size
        self.enemy_widget = enemy_widget
        self.training = training
        self.model = self._build_model()
        self.memory_feats = []
        self.memory_lbls = []
        if not training:
            K.set_learning_phase(0)

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
            [score_up, score_down]
        """
        inps = Input(shape=(10,))
        out = Dense(6, activation="elu")(inps)
        out = Dense(64, activation="elu")(out)
        out = Dense(32, activation="elu")(out)
        # out = Dense(1, activation="sigmoid")(out)
        out = Dense(1, activation="sigmoid")(out)
        model = Model(inputs=inps, outputs=out)
        if not os.path.isdir(self.save_path):
            model.save_weights(self.save_path)

        else:
            l.rl_loss = self.rl_loss
            model.load_weights(self.save_path)

        if self.training:
            model.compile(loss=self.rl_loss, optimizer=RMSprop(learning_rate=0.05))

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
        self.widget.center_y = int(
            self._decide(dt, ball_x, ball_y, ball_vel_x, ball_vel_y, own_x, own_y, enemy_x, enemy_y,
                         widget_size_y) * self.game_size[1]
        )
        if self.widget.center_y < 0:
            self.widget.center_y = 0
        elif self.widget.center_y > self.game_size[1]:
            self.widget.center_y = self.game_size[1]

    def _decide(self, dt, ball_x, ball_y, ball_vel_x, ball_vel_y, own_x, own_y, enemy_x, enemy_y, widget_size_y):
        """
        :return: new relative y betweem 0 (top) and 1 (bottom)
        """
        X = np.array([[dt, ball_x, ball_y, ball_vel_x, ball_vel_y, own_x, own_y, enemy_x, enemy_y, widget_size_y]])
        self.memory_feats.append(X)
        out = np.squeeze(self.model(X))
        assert 0 <= out <= 1, "this should't happen"
        return out

    def notify_end(self, won):
        self.memory_lbls += [float(not won)] * (len(self.memory_feats) - len(self.memory_lbls))

    def train(self):
        print("training started!")
        X = np.array(np.squeeze(self.memory_feats))
        y = np.array(self.memory_lbls)
        self.model.fit(X, y, epochs=5, verbose=0, )
        self.model.save(self.save_path)

    @staticmethod
    @function
    def rl_loss(y_true, y_pred):
        return K.sum(y_true) * K.sum(y_pred) ** 0
