from kivy import Config
from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import ObjectProperty, NumericProperty, Clock
from kivy.uix.widget import Widget
import numpy as np

_kv_loaded = False


class PongGame(Widget):
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)
    p1_score = ObjectProperty(None)
    p2_score = ObjectProperty(None)

class PongBall(Widget):
    # Only here to give the widget a name
    pass


class PongPaddle(Widget):
    score = NumericProperty(0)


class GUIController(App):
    def __init__(self, cfg, human_callback, update_fun):
        super().__init__()
        self._rel_ball = cfg["positions"]["ball"]
        self._rel_player2 = cfg["positions"]["player2"]
        self._rel_player1 = cfg["positions"]["player1"]
        self.game = None
        self.human_input = human_callback
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down, on_key_up=self._on_keyboard_up)
        self.model_update_fun = update_fun

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down, on_key_up=self._on_keyboard_up)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == "w":
            go = 1
        elif keycode[1] == "s":
            go = -1
        else:
            return
        self.human_input(go)

    def _on_keyboard_up(self, keyboard, keycode):
        self.human_input(0)

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
        Clock.schedule_interval(self.model_update_fun, 1.0 / 240.0)
        return self.game

    def update(self, player_1_pos: np.ndarray, player_2_pos: np.ndarray, player_1_score: int,
               player_2_score: int, ball_pos: np.ndarray) -> None:
        self._set_player_pos(0, player_1_pos)
        self._set_player_pos(1, player_2_pos)
        self.set_scores([player_1_score, player_2_score])
        self.set_ball_pos(ball_pos)

    def set_ball_pos(self, new_pos: np.ndarray) -> None:
        self.game.ball.center_x = int(new_pos[0] * 800)
        self.game.ball.center_y = int(new_pos[1] * 600)

    def set_player_y(self, player_id: int, y: int) -> None:
        assert player_id in [0, 1], "set player_y: invalid player id"
        if player_id == 0:
            self.game.player1.center_y = int(y * 600)
        else:
            self.game.player2.center_y = int(y * 600)

    def _set_player_pos(self, player_id: int, xy: np.ndarray) -> None:
        assert player_id in [0, 1], "set player_y: invalid player id"
        if player_id == 0:
            self.game.player1.center_x = int(xy[0] * 800)
            self.game.player1.center_y = int(xy[1] * 600)
        else:
            self.game.player2.center_x = int(xy[0] * 800)
            self.game.player2.center_y = int(xy[1] * 600)

    def set_score(self, player_id, new_score) -> None:
        assert player_id in [0, 1], "set score: invalid player id"
        if player_id == 0:
            self.game.p1_score.text = str(new_score)
        else:
            self.game.p2_score.text = str(new_score)

    def set_scores(self, scores) -> None:
        assert len(scores) == 2, "set_scores: invalid array length"
        self.set_score(0, scores[0])
        self.set_score(1, scores[1])
