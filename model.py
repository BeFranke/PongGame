from typing import Dict, Callable

import numpy as np

from player import Player

"""
ALL VALUES IN THE MODEL CLASS ARE NORMED TO [LIMIT_LOW, LIMIT_HIGH]
"""
LIMIT_LOW = 0       # TODO: There is a bug that occurs when this is set below 0. Fix.
LIMIT_HIGH = 2**32  # Depending on the inner working of the compiler, this may already be too high TODO experiment

# size of the players' paddles
PADDLE_SIZE = np.array([25.0 / 800, 200.0 / 600]) * (LIMIT_HIGH - LIMIT_LOW)

# x coordinate that marks the half of the game field
HALF = (LIMIT_HIGH - LIMIT_LOW) / 2.0

# utillity functions for getting a [0, 1] value from game coordinates, and the other way around
norm = lambda x: (x - LIMIT_LOW) / (LIMIT_HIGH - LIMIT_LOW)
unnorm = lambda x: x * (LIMIT_HIGH - LIMIT_LOW) + LIMIT_LOW


class GameModel:
    def __init__(self, cfg: Dict, player1: Player, player2: Player):
        """
        model of the game
        :param cfg: dict with all config values
        :param player1: Player object for left player
        :param player2: Player object for right player
        """
        self.player1: Player = player1
        self.player2: Player = player2
        self.cfg: Dict = cfg
        self.player_1_pos: np.ndarray = unnorm(np.array(cfg["positions"]["player1"]))
        self.player_2_pos: np.ndarray = unnorm(np.array(cfg["positions"]["player2"]))
        self.speed_limit = unnorm(cfg["limits"]["max_speed_player"])
        self.ball_pos: np.ndarray = np.zeros(2)
        self.ball_vel: np.ndarray = np.zeros(2)
        self._reset_ball()
        self.player_1_score: int = 0
        self.player_2_score: int = 0
        self.gui_update: Callable = lambda *args: None
        self.won_callback: Callable = lambda *args: None
        self._human_state = 0
        self.speedup = self.cfg["limits"]["speedup"]
        self.path_res = self.cfg["limits"]["path_resolution"]

    def update(self, dt: float) -> None:
        """
        compute next frame
        :param dt: time delta elapsed since last frame in milliseconds
        :return True if game ended, else false
        """
        # print(f"delta is {dt}")
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
                    # change reflection angle based on where the ball hit the paddle
                    paddle = self.player_1_pos if self.ball_pos[0] < HALF else self.player_2_pos
                    delta_rel = (p[1] - paddle[1]) / (2 * PADDLE_SIZE[1])        # this is in [-0.25; 0.25]
                    self.ball_vel[1] += self.ball_vel[0] * delta_rel
                    self.ball_vel[0] -= self.ball_vel[0] * delta_rel
                    print(f"Player {int(self.ball_pos[0] > HALF) + 1} hit!")
                    self.player2.pong() if self.ball_pos[0] > HALF else self.player1.pong()
                    break

        # ball bounces on top/bottom
        if LIMIT_LOW > self.ball_pos[1]:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = LIMIT_LOW

        elif LIMIT_HIGH < self.ball_pos[1]:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = LIMIT_HIGH

        # score
        if not LIMIT_LOW < self.ball_pos[0] < LIMIT_HIGH:
            score_player = int(self.ball_pos[0] < HALF)
            self._score(score_player)

        new_player1_y = unnorm(
            self.player1.play(dt, norm(self.player_1_pos), norm(self.player_2_pos), norm(self.ball_pos),
                              norm(self.ball_vel)))
        new_player2_y = unnorm(self.player2.play(dt, norm(self.player_1_pos), norm(self.player_2_pos),
                                                 norm(self.ball_pos), norm(self.ball_vel)))

        # if player 1 is human, this is where key input gets included (then new_player1_y == self.player_1_pos)
        new_player1_y += self._human_state * self.speed_limit

        self.player_1_pos[1] = self._trim_to_valid(self.player_1_pos[1], new_player1_y)
        self.player_2_pos[1] = self._trim_to_valid(self.player_2_pos[1], new_player2_y)

        self.gui_update(norm(self.player_1_pos), norm(self.player_2_pos), self.player_1_score,
                        self.player_2_score, norm(self.ball_pos))

        # print(f"new ball_pos is [{self.ball_pos[0]}, {self.ball_pos[1]}], vel [{self.ball_vel[0], self.ball_vel[1]}]")

    def _score(self, player: int) -> None:
        """
        update scores and check for game over
        :param player: id of scoring player
        """
        if player == 0:
            self.player_1_score += 1
            print(f"Player 1 scored! Score: {self.player_1_score}:{self.player_2_score}")
        else:
            self.player_2_score += 1
            print(f"Player 2 scored! Score: {self.player_1_score}:{self.player_2_score}")

        if self.player_1_score == self.cfg["limits"]["points_to_win"]:
            print("--------------------PLAYER 1 WINS THE GAME-------------------------")
            self.player1.game_over(True)
            self.player2.game_over(False)
            self.won_callback(0)
        elif self.player_2_score == self.cfg["limits"]["points_to_win"]:
            print("--------------------PLAYER 2 WINS THE GAME-------------------------")
            self.player1.game_over(False)
            self.player2.game_over(True)
            self.won_callback(1)
        else:
            self.player1.score(not bool(player))
            self.player2.score(bool(player))
            self._reset_ball()

    def _reset_ball(self) -> None:
        """
        put the ball in the middle, assign a semi-random velocity to it
        """
        self.ball_pos = np.array(self.cfg["positions"]["ball"], dtype=np.float64) * (LIMIT_HIGH - LIMIT_LOW) + LIMIT_LOW
        self.ball_vel = np.array(self.cfg["positions"]["ball_vel"], dtype=np.float64)
        sign_x = np.random.choice([-1, 1])
        rand_y = (np.random.rand() - 0.5) * self.ball_vel[0]
        self.ball_vel[1] += rand_y
        self.ball_vel[0] *= sign_x
        self.ball_vel *= (LIMIT_HIGH - LIMIT_LOW) + LIMIT_LOW

    def human_input(self, state: int) -> None:
        """
        callback for GUI whenever the human user gives key input
        :param state: 1 or -1 depending on UP or DOWN action
        """
        self._human_state = state

    def _player_boundaries(self, id: int) -> np.ndarray:
        """
        :param id: player id
        :return: boundaries of given player as [x_min, x_max, y_min, y_max]
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

    def _trim_to_valid(self, old_y: float, new_y: float) -> float:
        """
        checks if the given movement from old_y to new_y is under the speed limit in the game config
        if not, clips it to the speed limit
        :param old_y: previous y position
        :param new_y: desired y position
        :return: the real achieved y position. Is equal to new_y if the move was valid
        """
        if new_y - old_y > self.speed_limit:
            new_y = old_y + self.speed_limit
        elif new_y - old_y < - self.speed_limit:
            new_y = old_y - self.speed_limit
        if new_y < LIMIT_LOW:
            new_y = LIMIT_LOW
        elif new_y > LIMIT_HIGH:
            new_y = LIMIT_HIGH
        return new_y

    def reset(self) -> None:
        """
        completely resets the game
        """
        self.player_1_score = 0
        self.player_2_score = 0
        self.player_1_pos: np.ndarray = unnorm(np.array(self.cfg["positions"]["player1"]))
        self.player_2_pos: np.ndarray = unnorm(np.array(self.cfg["positions"]["player2"]))
        self._reset_ball()
