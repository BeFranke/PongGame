from random import randint
from typing import Union, List

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.properties import NumericProperty, ReferenceListProperty, \
    ObjectProperty, StringProperty
from kivy.uix.widget import Widget
from kivy.vector import Vector

from AI import *
from settings import Config


class PongGame(Widget):
    enabled = True
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)
    game_msg = StringProperty("")
    game_msg_expl = StringProperty("")
    SCORE_TO_WIN = Config.get('points_to_win')

    def __init__(self):
        super().__init__()
        player = Config.get('players')
        self.player_list: List[Union[AI, Human, None]] = [None] * 2
        self.player1widget = self.ids.player_1
        self.player2widget = self.ids.player_2
        for i in range(2):
            my_widget = self.player1widget if i == 0 else self.player2widget
            enemy_widget = self.player1widget if i == 1 else self.player2widget
            if player[i] == "Human":
                self.player_list[i] = Human(my_widget)
            elif isinstance(player[i], list):
                if player[i][0] == "Heuristic":
                    self.player_list[i] = Heuristic(player[i][1], my_widget, self.ball)
                elif player[i][0] == "NeuralNet":
                    self.player_list[i] = NeuralNet(player[i][1], my_widget, enemy_widget, self.ball)
                else:
                    raise Exception("Config Error: Malformatted AI entry!")
            else:
                raise Exception("Config Error: AI Type not understood!")

    def clear_game(self):
        self.game_msg = ""
        self.game_msg_expl = ""
        self.player1.score = 0
        self.player2.score = 0

    def on_touch_down(self, touch):
        if not self.enabled:
            self.clear_game()
            self.enabled = True

    def serve_ball(self):
        self.ball.center = self.center
        self.ball.velocity = Vector(5, 0).rotate(randint(0, 360))
        if abs(self.ball.velocity[0]) < abs(self.ball.velocity[1]):
            self.ball.velocity = \
                Vector(self.ball.velocity[1], self.ball.velocity[0])

    def game_end(self, winner):
        self.game_msg = "Player {} wins the game!".format(winner)
        self.game_msg_expl = "Tap to play again"
        self.enabled = False

    def update(self, dt):
        for player in self.player_list:
            if isinstance(player, AI):
                player.play(dt)

        if not self.enabled:
            return

        self.ball.move()

        self.player1.bounce_ball(self.ball)
        self.player2.bounce_ball(self.ball)

        if (self.ball.y < 0) or (self.ball.top > self.height):
            self.ball.velocity_y *= -1

        elif self.ball.x < self.x:
            self.player2.score += 1

            if self.player2.score == self.SCORE_TO_WIN:
                self.game_end(2)

            self.serve_ball()

        elif self.ball.x > self.width:
            self.player1.score += 1

            if self.player1.score == self.SCORE_TO_WIN:
                self.game_end(1)

            self.serve_ball()


class PongBall(Widget):
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)

    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos


class PlayerWidget(Widget):
    score = NumericProperty(0)

    # pong_sound = SoundLoader.load('./res/pong.wav')

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            # self.pong_sound.play()
            randomness = random() * 0.4 + 0.8
            speedup = Config.get('speedup') * randomness
            offset = Config.get('offset') * \
                     Vector(0, ball.center_y - self.center_y)
            ball.velocity = speedup * \
                            Vector(-ball.velocity_x, ball.velocity_y) + offset


class Human:
    def __init__(self, widget):
        self.widget = widget
        widget.on_touch_move = self.on_touch_move

    def on_touch_move(self, touch):
        if touch.x < self.widget.width / 3:
            self.widget.center_y = touch.y
        elif touch.x > self.widget.width - self.widget.width / 3:
            self.widget.center_y = touch.y

    def on_pong(self):
        pass


class PongApp(App):
    def build(self):
        game = PongGame()
        game.serve_ball()
        Clock.schedule_interval(game.update, 1.0 / float(Config.get('frame_limit')))
        return game


if __name__ == "__main__":
    Config.load()
    # Window.fullscreen = 'auto'
    PongApp().run()
