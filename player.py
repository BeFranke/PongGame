import os
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import tensorflow.keras as K
from typing import Deque, List, Tuple


class Player(ABC):
    """
    generic player base class
    ensures all subclasses have an id-attribute and a play-method
    """

    def __init__(self, id):
        """
        :param id: the player id, either 0 (left) or 1 (right)
        """
        self.id = id

    @abstractmethod
    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray) -> float:
        """
        generic play method. Take in game state and return desired new y-position. All positions can be assumed to be
        normed to [0, 1]
        :param dt: time delta in milliseconds
        :param player1_pos: position of player 1 paddle as [x, y]
        :param player2_pos: position of player 2 paddle as [x, y
        :param ball_pos: position of pong ball as [x, y]
        :param ball_vel: velocity of pong ball as [x_vel, y_vel]
        :return: new y coordinate in [0, 1].
        If the returned position is too far away from the last position (speed limit in config file),
        the game will only update the position to the allowed extent
        """
        pass

    @abstractmethod
    def pong(self):
        """
        called whenever the player hits the ball
        """
        pass

    @abstractmethod
    def score(self, you_scored: bool):
        """
        called when ever someone scored
        :param you_scored: True if this player scored
        """
        pass

    @abstractmethod
    def game_over(self, won: bool):
        """
        called when ever someone wins
        :param won: True if this player won
        """
        pass

class Dummy(Player):
    """
    The simplest type of Pong AI:
    just follow the ball vertically
    Can be beaten by exploiting the game's player-speed limit
    """

    def __init__(self, id):
        super().__init__(id)

    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray) -> float:
        return ball_pos[1]

    def pong(self):
        pass

    def score(self, you_scored: bool):
        pass

    def game_over(self, won: bool):
        pass


class NeuralNet(Player):
    """
    tf.keras based deep-Q agent
    """
    def __init__(self, id, speed_limit: float, model_path: str = "models/DeepQ", training: bool = True,
                 gamma: float = 0.1, epsilon: float = 0.1, epsilon_decay: float = 0.99, pong_reward: int = 1,
                 win_reward: int = 20):
        super().__init__(id)
        self.win_reward = win_reward
        self.pong_reward = pong_reward
        self.speed_limit = speed_limit
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.model: K.Model = self._create_or_load()
        self.model_path: str = model_path
        self.training: bool = training
        self._memory: Deque = deque(maxlen=100000)
        self.gamma: float = gamma
        self.actions: List[int] = [1, 0, -1]
        # list of (state, action, reward, next_state, done) tuples
        self.last_state = {
            "state": None,
            "action": None,
            "reward": None,
            "next_state": None,
            "done": None
        }

    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray) -> float:

        # compute the state-vector
        if self.id == 0:
            state = np.array([dt, self.id, player1_pos[0], player1_pos[1], player2_pos[0], player2_pos[1],
                              ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1]])
        else:
            # mirror the field to make "finding itself" easier for the NeuralNet
            state = np.array([dt, self.id, - player2_pos[0] + 1, player2_pos[1], - player1_pos[0] + 1,
                              player1_pos[1], - ball_pos[0] + 1, ball_pos[1], - ball_vel[0] + 1, ball_vel[1]])

        # manage memory
        # TODO: do not overwrite if last action was succesful
        if self.last_state["state"] is not None and self.last_state["action"] is not None:
            self.last_state["reward"] = 0
            self.last_state["next_state"] = state
            self.last_state["done"] = False

        self._memory.append(self.last_state)

        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(self.model.predict(state))
        return action * self.speed_limit

    def pong(self):
        pass

    def score(self, you_scored: bool):
        pass

    def game_over(self, won: bool):
        pass

    def _create_or_load(self):
        # all features need to be normalized respective to the player
        # features: dt, my_x, my_y, enemy_x, enemy_y, ball_x, ball_y, ball_vel_x, ball_vel_y
        inps = K.layers.Input(shape=(9,))
        x = K.layers.Dense(128, activation="selu")(inps)
        x = K.layers.Dense(256, activation="selu")(x)
        x = K.layers.Dense(64, activation="selu")(x)
        x = K.layers.Dense(3, activation="linear")(x)
        model = K.Model(inputs=inps, outputs=x)
        if os.path.isdir(self.model_path):
            model.load_weights(self.model_path)
        model.compile(loas="MSE", optimizer="adam")
        return model


class Human(Player):
    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray) -> float:
        # only here to make interaction easier
        return player1_pos[1] if self.id == 0 else player2_pos[1]

    def __init__(self, id):
        super().__init__(id)

    def pong(self):
        pass

    def score(self, you_scored: bool):
        pass

    def game_over(self, won: bool):
        pass


class Heuristic(Player):
    """
    TODO
    AI that pre-computes the trajectory of the ball, moves there ASAP
    and has an given probabillity of making errors of some sort
    """

    def __init__(self, id):
        super().__init__(id)
        raise NotImplementedError

    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray) -> float:
        pass

    def pong(self):
        pass

    def score(self, you_scored: bool):
        pass

    def game_over(self, won: bool):
        pass
