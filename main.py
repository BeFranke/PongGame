from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty, StringProperty
from kivy.vector import Vector
from kivy.clock import Clock
from random import randint, choice
from kivy.core.window import Window
from AI import *
from settings import Config

class PongGame(Widget):
    enabled = True
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)
    game_msg = StringProperty("")
    game_msg_expl = StringProperty("")
    SCORE_TO_WIN = Config.get('points to win')

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
        self.game_msg_expl = "Tap to play again!"
        self.enabled = False

    def update(self, dt):
        if not self.enabled:
            return

        self.ball.move()
        self.player2.play(dt, self.ball.velocity, self.ball.pos)

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

class Player(Widget):

    score = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            speedup  = 1.1
            x_speedup = 0 if ball.velocity_x > ball.velocity_y \
                    else choice(range(0, 5)) * Config.get('speedup')
            offset = Config.get('offset') * \
                Vector(x_speedup, ball.center_y-self.center_y)
            ball.velocity =  speedup * \
                Vector(-ball.velocity_x, ball.velocity_y) + offset

class Human(Player):
    def on_touch_move(self, touch):
        if touch.x < self.width/3:
            self.center_y = touch.y
        elif touch.x > self.width - self.width/3:
            self.center_y = touch.y

class AI(Player):
    ai = Config.get('AI')
    if isinstance(ai, tuple):
        if ai[0] == "Heuristic":
            decisionMaker = Heuristic(ai[1])
    else:
        raise Exception("Config Error: AI Type not understood!")
        exit(1)

    def play(self, dt, ball_vel, ball_pos):
        delta_y = self.decisionMaker.decide(dt, ball_vel, ball_pos,\
                        Window.size, self.size, self.center_y)
        self.center_y += delta_y if abs(delta_y) > 1 else 0

class PongApp(App):
    def build(self):
        game = PongGame()
        game.serve_ball()
        Clock.schedule_interval(game.update, 1.0/float(Config.get('frame limit')))
        return game

if __name__ == "__main__":
    Config.load()
    PongApp().run()
