from random import choice, random

class Legacy:
    def __init__(self):
        self.direction = choice([-1, 1])

    def decide(self, dt, ball_vel, ball_pos, window_dims, paddle_size, center_y):

        speed = random() * 0.8 + 1

        delta_y = self.direction * window_dims[1] * speed/(dt*10000)
        if center_y - paddle_size[1]/2 <= 0 and self.direction < 0 \
            or center_y + paddle_size[1]/2 >= window_dims[1] and self.direction > 0:
            self.direction *= -1

        return delta_y

class Heuristic:
    def __init__(self, speed_limit):
        # speed limit is given in height/sec
        self.speed_limit = speed_limit
        self.rand = random()

    def on_pong(self):
        self.rand = random()

    def decide(self, dt, ball_vel, ball_pos, window_dims, paddle_size, center_y):
        desired = ball_pos[1] - center_y
        #error_margin = (ball_vel[1]/abs(ball_vel[1])) * paddle_size[1] * 0.1 * self.rand * 3
        #desired += error_margin
        if self.speed_limit == -1 or abs(desired/dt) <= self.speed_limit:
            return desired
        else:
            return (self.speed_limit * desired / abs(desired)) * dt
