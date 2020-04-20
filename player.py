from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Dict, Union

import numpy as np
import tensorflow.keras as K

State = Dict[str, Union[np.ndarray, int, bool, None]]
STATE_DIM = 6

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
    def __init__(self, id, speed_limit: float, model_path: str = "models/DeepPongQ", training: bool = True,
                 gamma: float = 0.99, epsilon: float = 0.7, epsilon_decay: float = 0.99, pong_reward: int = 0,
                 win_reward: int = 1, epsilon_min: float = 0.001, batch_size=2048, checkpoints=True):
        super().__init__(id)
        self.checkpoints = checkpoints
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.win_reward = win_reward
        self.pong_reward = pong_reward
        self.speed_limit = speed_limit
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.model_path: str = model_path
        self.training: bool = training
        self._memory: Deque[State] = deque(maxlen=100000)
        self.gamma: float = gamma
        self.actions: List[int] = [1, 0, -1]
        # list of (state, action, reward, next_state, done) tuples
        self.last_state: State = {
            "state": None,
            "action": None,
            "reward": None,
            "next_state": None,
            "done": None
        }
        self.model: K.Model = self._create_or_load()
        self.target_model: K.Model = self._create_or_load()

    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray) -> float:

        # compute the state-vector
        if self.id == 0:
            my_pos = player1_pos
            state = np.array([[player1_pos[1], player2_pos[1], ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1]]])
        else:
            # mirror the field to make "finding itself" easier for the NeuralNet
            my_pos = player2_pos
            state = np.array([[player2_pos[1], player1_pos[1], - ball_pos[0] + 1, ball_pos[1],
                               - ball_vel[0], ball_vel[1]]])

        # manage memory
        # if last state exists, but reward is none, the last action did not yield any reward
        if self.training:
            if self.last_state["state"] is not None:
                if self.last_state["reward"] is None:
                    self.last_state["reward"] = 0
                self.last_state["next_state"] = state
                self.last_state["done"] = False
                # only save 1/10 th of uninteresting states to reduce correlation
                if np.random.rand() < 0.1:
                    self._memory.append(self.last_state)
            self._state_reset()
            self.last_state["state"] = state
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.actions[int(np.argmax(self.model(state)))]
        self.last_state["action"] = action
        return my_pos[1] + action * self.speed_limit

    def train(self):
        X = np.zeros((len(self._memory), 6))
        y = np.zeros((len(self._memory), 3))
        for i, d in enumerate(self._memory):
            state = d["state"]
            action = d["action"]
            reward = d["reward"]
            done = d["done"]
            next_state = d["next_state"]

            y_target = self.target_model(state).numpy()
            y_target[0][action] = reward if done \
                else reward + self.gamma * np.max(np.asarray(self.target_model(next_state)[0]))
            X[i][:] = state[0][:]
            y[i][:] = y_target[0][:]

        self.model.fit(X, y, batch_size=self.batch_size, verbose=2, epochs=50, shuffle=True)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.save() if self.checkpoints else None

    def save(self):
        self.model.save_weights(self.model_path)

    def pong(self):
        self.last_state["reward"] = self.pong_reward

    def score(self, you_scored: bool):
        sign = 1 if you_scored else -1
        self.last_state["reward"] = self.win_reward * sign
        self.last_state["done"] = True
        self._memory.append(self.last_state)
        self._state_reset()

    def game_over(self, won: bool):
        self.score(won)

    def reload_from_path(self, path):
        tmp = self.model_path
        self.model_path = path
        self.model = self._create_or_load()
        self.model_path = tmp

    def update_target_model(self):
        self.target_model.load_weights(self.model_path)

    def _create_or_load(self):
        # all features need to be normalized respective to the player
        # features: my_y, enemy_y, ball_x, ball_y, ball_vel_x, ball_vel_y
        inps = K.layers.Input(shape=(STATE_DIM,))
        # the actual hidden layers
        # deep version
        """
        x = K.layers.Dense(128, activation="elu")(inps)
        x = K.layers.Dense(256, activation="elu")(x)
        x = K.layers.Dense(512, activation="elu")(x)
        x = K.layers.Dense(256, activation="elu")(x)
        """
        x = K.layers.Dense(48, activation="tanh")(inps)
        x = K.layers.Dense(22, activation="tanh")(x)
        # output
        x = K.layers.Dense(3, activation="linear")(x)
        model = K.Model(inputs=inps, outputs=x)
        try:
            model.load_weights(self.model_path)
            print("loading model...")
        except:
            print("No model found, creating new one..")
        if self.training:
            model.compile(loss="MSE", optimizer="rmsprop")
        return model

    def _state_reset(self):
        self.last_state = {
            "state": None,
            "action": None,
            "reward": None,
            "next_state": None,
            "done": None
        }


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

class Classic(Player):
    """
    Just moves up and down
    """

    def __init__(self, id):
        super().__init__(id)
        self.state = 1

    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray) -> float:
        my_pos = player1_pos[1] if self.id == 0 else player2_pos[1]
        if my_pos > 0.8 or my_pos < 0.2:
            self.state *= -1
        return my_pos + self.state

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
