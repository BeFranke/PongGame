import copy
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Dict, Tuple, Union

import numpy as np
import torch
import torchvision as tv
from torchvision import transforms as tr
from tqdm import tqdm
from PIL import Image


STATE_DIM = 6
TRAIN_EPOCHS = 100
MEMORY_LIMIT = 10000
N_TIMESTEPS = 4

SCREENSHOT_PROBA = 0.0

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
             ball_vel: np.ndarray, state: Image) -> float:
        """from matplotlib import pyplot as plt
        generic play method. Take in game state and return desired new y-position. All positions can be assumed to be
        normed to [0, 1]
        :param dt: time delta in milliseconds
        :param player1_pos: position of player 1 paddle as [x, y]
        :param player2_pos: position of player 2 paddle as [x, y
        :param ball_pos: position of pong ball as [x, y]
        :param ball_vel: velocity of pong ball as [x_vel, y_vel]
        :param state: Canvas as PIL.Image
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
             ball_vel: np.ndarray, state: Image) -> float:
        return ball_pos[1]

    def pong(self):
        pass

    def score(self, you_scored: bool):
        pass

    def game_over(self, won: bool):
        pass


@dataclass
class ReplayMemory:
    state: torch.tensor     # shape (n_timesteps, c, h, w)
    action: int
    next_state: torch.tensor
    reward: float
    done: bool


class NeuralNet(Player):
    """
    torch-based deep-Q agent
    """
    def __init__(self, id, speed_limit: float, model_path: str = "models/DeepPongQ", training: bool = True,
                 gamma: float = 0.99, epsilon: float = 0.99, epsilon_decay: float = 0.9, pong_reward: float = 0.01,
                 win_reward: float = 1.0, epsilon_min: float = 0.001, batch_size=2048, checkpoints=True):
        super().__init__(id)
        # hyperparameters
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
        self.gamma: float = gamma
        self.actions: List[int] = [1, 0, -1]
        self.model, self.optimizer, self.loss = self._create_or_load()
        self.target_model = copy.deepcopy(self.model)
        self.as_torch = tr.Compose([
            tr.Resize(size=(160, 240)),
            tr.ToTensor()
        ])

        # memory
        self.memory: Deque[ReplayMemory] = deque(maxlen=1000)
        self.last_state: ReplayMemory = None

        self.model.eval()
        self.target_model.eval()

    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray, state: Image) -> float:

        if (w := np.random.random()) < SCREENSHOT_PROBA:
            state.save(f"debug/{int(w * 100000)}.png")

        my_pos = player1_pos if self.id == 0 else player2_pos
        state = self._preprocess_state(state)

        if self.training:
            # prepare new state
            if self.last_state is not None and self.last_state.state is not None:
                if not self.last_state.done:
                    # last state was not terminal -> reward 0 and save
                    if self.last_state.reward is None:
                        self.last_state.reward = 0
                        
                    self.memory.append(copy.deepcopy(self.last_state))

                temp = self.last_state.state[:, :, :]
                self.last_state.state = torch.empty_like(temp)
                self.last_state.state[:N_TIMESTEPS-1, :, :] = temp[1:, :, :]
            else:
                self.last_state = ReplayMemory(torch.zeros(N_TIMESTEPS, *state.shape), None, None, None, None)

            self.last_state.state[-1, :, :] = state

            if self.memory:
                self.memory[-1].next_state = self.last_state.state


            if np.random.rand() <= self.epsilon:
                action = np.random.choice(self.actions)
            else:
                action = self.actions[int(torch.argmax(self.model(self.last_state.state[None, :, :, :])))]
        else:
            action = self.actions[int(torch.argmax(self.model(self.last_state.state[None, :, :, :])))]

        self.last_state.action = action
        return my_pos[1] + action * self.speed_limit


    def train(self):
        self.model.train()
        self.target_model.eval()
        X = torch.zeros((len(self.memory), *self.memory[0].state.shape))
        y = torch.zeros((len(self.memory), 3))
        print("Post-processing memory...")
        for i, e in tqdm(enumerate(self.memory), total=len(self.memory)):
            state = e.state
            action = e.action
            reward = e.reward
            done = e.done
            next_state = e.next_state

            y_target = self.target_model(state[None, :, :, :])
            y_target[0][action] = reward if done \
                else reward + self.gamma * torch.max(self.target_model(next_state[None, :, :, :])[0].detach())
            X[i] = state
            y[i] = y_target[0]

        # decorrelate data
        inds = torch.randperm(len(X))
        X, y = X[inds], y[inds]

        with tqdm(range(TRAIN_EPOCHS), desc="Loss: 0.00", total=TRAIN_EPOCHS) as pbar:
            print("Training start!")
            for epoch in range(TRAIN_EPOCHS):
                for batch_inds in torch.split(torch.arange(len(X)), 128):
                    pred = self.model(X[inds])
                    loss = self.loss(pred, y[inds])
                    loss.backward(retain_graph=True)
                    pbar.set_description(f"Loss: {loss.item():.2f}")
                    self.optimizer.step()
                pbar.update(1)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.checkpoints:
            self.save()

        self.model.eval()

    def save(self):
        torch.save(self.model, self.model_path)

    def pong(self):
        self.last_state.reward = self.pong_reward
        self.memory.append(self.last_state)

    def score(self, you_scored: bool):
        sign = 1 if you_scored else -1
        self.last_state.reward = self.win_reward * sign
        self.last_state.done = True
        self.memory.append(self.last_state)

    def game_over(self, won: bool):
        self.score(won)

    def reload_from_path(self, path):
        tmp = self.model_path
        self.model_path = path
        self.model, _, _ = self._create_or_load()
        self.model_path = tmp

    def update_target_model(self):
        # self.target_model.load_weights(self.model_path)
        self.target_model = torch.load(self.model_path)

    def _create_or_load(self) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.nn.Module]:
        nn = torch.nn
        
        try:
            # model.load_weights(self.model_path)
            model = torch.load(self.model_path)
            print("loading model...")
        except FileNotFoundError:
            print("No model found, creating new one..")
            model = tv.models.mobilenet_v3_small()
            model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
            model.features[0][0] = torch.nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        loss = nn.HuberLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        model.eval()
        return model, optimizer, loss

    def _preprocess_state(self, state: Image):
        # convert to torch
        state = self.as_torch(state)[:4].mean(dim=0)

        # if not player 1, mirror along x
        if self.id != 0:
            state = torch.flip(state, dims=(0,))

        return state


class Human(Player):
    def play(self, dt: float, player1_pos: np.ndarray, player2_pos: np.ndarray, ball_pos: np.ndarray,
             ball_vel: np.ndarray, _) -> float:
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
             ball_vel: np.ndarray, _) -> float:
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
