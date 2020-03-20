from kivy import Config
from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import ObjectProperty, NumericProperty
from kivy.uix.widget import Widget
import numpy as np

_kv_loaded = False


class PongGame(Widget):
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)


class PongBall(Widget):
    # Only here to give the widget a name
    pass


class PongPaddle(Widget):
    score = NumericProperty(0)


class GUIController(App):

    def __init__(self, rel_player1: np.ndarray, rel_player2: np.ndarray, rel_ball: np.ndarray):
        super().__init__()
        self._rel_ball = rel_ball
        self._rel_player2 = rel_player2
        self._rel_player1 = rel_player1
        self.game = None

    def build(self):
        if not _kv_loaded:
            Builder.load_file("ui.kv")

        Config.set('graphics', 'height', 600)
        Config.set('graphics', 'width', 800)
        Config.set('graphics', 'resizable', False)
        Config.write()
        self.game = PongGame()
        self.set_ball_pos(self._rel_ball)
        self._set_player_pos(0, self._rel_player1)
        self._set_player_pos(1, self._rel_player2)
        return self.game

    def set_ball_pos(self, new_pos: np.ndarray):
        self.game.ball.center_x = int(new_pos[0] * Window.width)
        self.game.ball.center_y = int(new_pos[1] * Window.height)

    def set_player_y(self, player_id: int, y: int):
        assert player_id in [0, 1], "set player_y: invalid player id"
        if player_id == 0:
            self.game.player1.center_y = int(y * Window.height)
        else:
            self.game.player2.center_y = int(y * Window.height)

    def _set_player_pos(self, player_id: int, xy: np.ndarray):
        assert player_id in [0, 1], "set player_y: invalid player id"
        if player_id == 0:
            self.game.player1.center_x = int(xy[0] * Window.width)
            self.game.player1.center_y = int(xy[1] * Window.height)
        else:
            self.game.player2.center_x = int(xy[0] * Window.width)
            self.game.player2.center_y = int(xy[1] * Window.height)

    def set_score(self, player_id, new_score):
        assert player_id in [0, 1], "set score: invalid player id"
        if player_id == 0:
            self.game.player1.score = new_score
        else:
            self.game.player2.score = new_score

    def set_scores(self, scores):
        assert len(scores) == 2, "set_scores: invalid array length"
        self.game.player1.score, self.game.player2.score = scores
