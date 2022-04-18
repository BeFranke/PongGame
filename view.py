from typing import Dict, Callable, List
from io import BytesIO

from kivy import Config
from kivy.app import App
from kivy.core.window import Window
from kivy.graphics import Color
from kivy.lang import Builder
from kivy.properties import ObjectProperty, NumericProperty, Clock, ListProperty
from kivy.uix.widget import Widget

import numpy as np
from PIL import Image



_kv_loaded = False


class PongGame(Widget):
    """
    Game field
    """
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)
    p1_score = ObjectProperty(None)
    p2_score = ObjectProperty(None)


class PongBall(Widget):
    """
    Game ball
    Only here to give the widget a name, for better readability
    """
    pass


class PongPaddle(Widget):
    """
    Players' paddles
    """
    score = NumericProperty(0)
    rgb = ListProperty([1,1,1])


class GUIController(App):
    def __init__(self, cfg: Dict, human_callback: Callable[[int], None],
                 update_fun: Callable[[float], None], resume_fun: Callable[[], None]):
        """
        This is the "View" class of the MVC pattern.
        It updates the GUI according to the information it gets from the model
        It also passes the user input to the model
        IMPORTANT: All position-arguments passed to methods of this class are assumed to be in the range [0, 1] and
                    adapted to screen resolution internally
        :param cfg: Game config dictionary
        :param human_callback: function that accepts an int, either -1, 0 or 1,
                        representing the user input UP, STAY or DOWN
        :param update_fun: The function that updates the model, will be scheduled here with kivy tools if GUI is enabled
                            Yes, this violates the MVC pattern
        :param resume_fun: The function to be called when a new game is requested by the user
        """
        super().__init__()
        self._rel_ball = cfg["positions"]["ball"]
        self._rel_player2 = cfg["positions"]["player2"]
        self._rel_player1 = cfg["positions"]["player1"]
        self.game = None
        self.human_input = human_callback
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self.root)
        self._keyboard.bind(on_key_down=self._on_keyboard_down, on_key_up=self._on_keyboard_up)
        self.model_update_fun = update_fun
        self.resume_fun = resume_fun
        self.go = True
        self.loop_event = None
        self.cfg = cfg

    def _keyboard_closed(self):
        """
        cleans up the key binding on exit
        """
        self._keyboard.unbind(on_key_down=self._on_keyboard_down, on_key_up=self._on_keyboard_up)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        """
        parses user input when user presses a key
        :param keyboard: ignored
        :param keycode: the key that was pressed
        :param text: ignored
        :param modifiers: ignored
        """
        if keycode[1] == "enter" and not self.go:
            self.go = True
            self.game.lbl_game_msg.text = ""
            self.game.p1_score.text = "0"
            self.game.p2_score.text = "0"
            self.resume_fun()
            Clock.schedule_interval(self.model_update_fun, 1.0 / 240.0)
            return
        elif keycode[1] == "w":
            go = 1
        elif keycode[1] == "s":
            go = -1
        else:
            return
        self.human_input(go)

    def _on_keyboard_up(self, keyboard, keycode):
        """
        user has let go of a key
        :param keyboard: ignored
        :param keycode: ignored
        :return:
        """
        self.human_input(0)

    def build(self):
        """
        kivy-build function that constructs the GUI
        """
        if not _kv_loaded:
            Builder.load_file("ui.kv")

        # currently fixed 800x600 resolution, because this is Pong, not Skyrim
        Config.set('graphics', 'height', self.cfg["resolution"][1])
        Config.set('graphics', 'width', self.cfg["resolution"][0])
        Config.set('graphics', 'resizable', False)
        Config.write()
        self.game: PongGame = PongGame()
        self.game.player1.background_color = (168, 78, 50)
        self.game.player1.background_color = (50, 86, 168)
        self._set_ball_pos(self._rel_ball)
        self._set_player_pos(0, self._rel_player1)
        self._set_player_pos(1, self._rel_player2)
        self.loop_event = Clock.schedule_interval(self.model_update_fun, 1.0 / 120.0)
        return self.game

    def update(self, player_1_pos: np.ndarray, player_2_pos: np.ndarray, player_1_score: int,
               player_2_score: int, ball_pos: np.ndarray) -> Image:
        """
        updates the GUI
        :param player_1_pos: position of player 1 as [x, y]
        :param player_2_pos: position of player 2 as [x, y]
        :param player_1_score: score of player 1
        :param player_2_score: score of player 2
        :param ball_pos: position of the ball as [x, y]

        :return PIL.Image of the canvas
        """
        self._set_player_pos(0, player_1_pos)
        self._set_player_pos(1, player_2_pos)
        self._set_scores([player_1_score, player_2_score])
        self._set_ball_pos(ball_pos)

        tfile = BytesIO()
        self.game.export_as_image().save(tfile, fmt="png")

        return Image.open(tfile)

    def _set_ball_pos(self, new_pos: np.ndarray) -> None:
        """
        moves the ball to a new position
        :param new_pos: new position as [x, y]
        """
        self.game.ball.center_x = int(new_pos[0] * self.cfg["resolution"][0])
        self.game.ball.center_y = int(new_pos[1] * self.cfg["resolution"][1])

    def _set_player_pos(self, player_id: int, xy: np.ndarray) -> None:
        """
        moves player to position
        :param player_id: 0 or 1 for left or right player
        :param xy: new position as [x, y]
        """
        assert player_id in [0, 1], "set player_y: invalid player id"
        if player_id == 0:
            self.game.player1.center_x = int(xy[0] * self.cfg["resolution"][0])
            self.game.player1.center_y = int(xy[1] * self.cfg["resolution"][1])
            self.game.player1.rgb = (1,0,0)
        else:
            self.game.player2.center_x = int(xy[0] * self.cfg["resolution"][0])
            self.game.player2.center_y = int(xy[1] * self.cfg["resolution"][1])

    def _set_score(self, player_id: int, new_score: int) -> None:
        """
        sets the score of a single player
        :param player_id: 0 or 1 for left or right player
        :param new_score: the score to be displayed henceforth
        """
        assert player_id in [0, 1], "set score: invalid player id"
        if player_id == 0:
            self.game.p1_score.text = str(new_score)
        else:
            self.game.p2_score.text = str(new_score)

    def _set_scores(self, scores: List[int]) -> None:
        """
        sets the scores of both players
        :param scores: new scores as [left_player_score, right_player_score]
        """
        assert len(scores) == 2, "set_scores: invalid array length"
        self._set_score(0, scores[0])
        self._set_score(1, scores[1])

    def game_won(self, pid: int):
        """
        callback for model if game was won
        :param pid: 0 or 1 for left or right player
        """
        self.loop_event.cancel()
        self.game.lbl_game_msg.text = f"Player {pid + 1} won! \n ENTER to play again, ESC to exit"
        self.go = False
