from typing import Dict, Callable

import numpy as np

from player import Player

"""
ALL VALUES IN THE MODEL CLASS ARE NORMED TO [LIMIT_LOW, LIMIT_HIGH]
"""
LIMIT_LOW = 0
LIMIT_HIGH = 10
PADDLE_SIZE = np.array([25.0 / 800, 200.0 / 600]) * (LIMIT_HIGH - LIMIT_LOW)
HALF = (LIMIT_HIGH - LIMIT_LOW) / 2.0

norm = lambda x: (x - LIMIT_LOW) / (LIMIT_HIGH - LIMIT_LOW)
unnorm = lambda x: x * (LIMIT_HIGH - LIMIT_LOW) + LIMIT_LOW


class GameModel:
    def __init__(self, cfg: Dict, player1: Player, player2: Player):
        self.player1: Player = player1
        self.player2: Player = player2
        self.cfg: Dict = cfg
        self.player_1_pos: np.ndarray = unnorm(np.array(cfg["positions"]["player1"]))
        self.player_2_pos: np.ndarray = unnorm(np.array(cfg["positions"]["player2"]))
        self.speed_limit = unnorm(cfg["limits"]["max_speed_player"])
        self.ball_pos: np.ndarray = np.zeros(2)
        self.ball_vel: np.ndarray = np.zeros(2)
        self.reset_ball()
        self.player_1_score: int = 0
        self.player_2_score: int = 0
        self.gui_update: Callable = lambda *args: None
        self._human_state = 0
        self.speedup = self.cfg["limits"]["speedup"]
        self.path_res = self.cfg["limits"]["path_resolution"]

    def update(self, dt: float) -> None:
        # move the ball
        last_pos = self.ball_pos.copy()
        self.ball_pos += self.ball_vel * dt
        # check for collisions intelligently
        if self.ball_pos[0] < HALF:
            bounds = self._player_boundaries(0)
        else:
            bounds = self._player_boundaries(1)

        x_bound = bounds[0] if self.ball_pos[0] > HALF else bounds[1]

        # player hits ball
        if (last_pos[0] <= x_bound <= self.ball_pos[0] or last_pos[0] >= x_bound >= self.ball_pos[0])\
                and np.sign(self.ball_pos[0] - last_pos[0]) == np.sign(self.ball_pos[0] - HALF):
            # print("TEST")
            # trajectory-based check
            # generate 100 points on the trajectory, collision check for each of them
            for i in np.linspace(0, 1, self.path_res):
                p = i * self.ball_pos + (1 - i) * last_pos
                if bounds[0] <= p[0] <= bounds[1] and bounds[2] <= p[1] <= bounds[3]:
                    # bounce ball + speed up
                    self.ball_vel[0] *= - 1
                    self.ball_vel *= self.speedup
                    print(f"Player {int(self.ball_pos[0] < HALF) + 1} hit!")
                    break

        # ball bounces on top/bottom
        elif not LIMIT_LOW < self.ball_pos[1] < LIMIT_HIGH:
            self.ball_vel[1] *= -1

        # score
        elif not LIMIT_LOW < self.ball_pos[0] < LIMIT_HIGH:
            score_player = int(self.ball_pos[0] < HALF)
            self._score(score_player)

        new_player1_y = unnorm(
            self.player1.play(dt, norm(self.player_1_pos), norm(self.player_2_pos), norm(self.ball_pos),
                              norm(self.ball_vel)))
        new_player2_y = unnorm(self.player2.play(dt, norm(self.player_1_pos), norm(self.player_2_pos),
                                                 norm(self.ball_pos), norm(self.ball_vel)))

        # if player 1 is human, this is where key input gets included (then new_player1_y == self.player_1_pos)
        new_player1_y += self._human_state * self.speed_limit

        self.player_1_pos[1] = self.trim_to_valid(self.player_1_pos[1], new_player1_y)
        self.player_2_pos[1] = self.trim_to_valid(self.player_2_pos[1], new_player2_y)

        self.gui_update(norm(self.player_1_pos), norm(self.player_2_pos), self.player_1_score,
                        self.player_2_score, norm(self.ball_pos))

        # print(f"new ball_pos is [{self.ball_pos[0]}, {self.ball_pos[1]}], vel [{self.ball_vel[0], self.ball_vel[1]}]")

    def _score(self, player: int) -> None:
        if player == 0:
            print("Player 1 scored!")
            self.player_1_score += 1
        else:
            print("Player 2 scored!")
            self.player_2_score += 1
        self.reset_ball()

    def reset_ball(self):
        self.ball_pos = np.array(self.cfg["positions"]["ball"], dtype=np.float64) * (LIMIT_HIGH - LIMIT_LOW) + LIMIT_LOW
        self.ball_vel = np.array(self.cfg["positions"]["ball_vel"], dtype=np.float64)
        sign_x = np.random.choice([-1, 1])
        rand_y = (np.random.rand() - 0.5) * self.ball_vel[0]
        self.ball_vel[1] += rand_y
        self.ball_vel[0] *= sign_x
        self.ball_vel *= (LIMIT_HIGH - LIMIT_LOW) + LIMIT_LOW

    def human_input(self, state: int) -> None:
        self._human_state = state

    def _player_boundaries(self, id: int) -> np.ndarray:
        """
        :param id: player id
        :return: boundaries as [x_min, x_max, y_min, y_max]
        """
        dy = PADDLE_SIZE[1] / 2
        dx = PADDLE_SIZE[0] / 2
        if id == 0:
            center = self.player_1_pos
        else:
            center = self.player_2_pos

        return np.array([
            center[0] - dx, center[0] + dx, center[1] - dy, center[1] + dy
        ])

    def trim_to_valid(self, old_y: float, new_y: float) -> float:
        if new_y - old_y > self.speed_limit:
            new_y = old_y + self.speed_limit
        elif new_y - old_y < - self.speed_limit:
            new_y = old_y - self.speed_limit
        if new_y < LIMIT_LOW:
            new_y = LIMIT_LOW
        elif new_y > LIMIT_HIGH:
            new_y = LIMIT_HIGH
        return new_y
