from random import choice, random

class Heuristic:
    def __init__(self):
        self.direction = choice([-1, 1])

    def decide(self, dt, ball_vel, ball_pos, window_dims, paddle_size, center_y):

        speed = random() * 0.8 + 1

        delta_y = self.direction * window_dims[1] * speed/(dt*10000)
        if center_y - paddle_size[1]/2 <= 0 and self.direction < 0 \
            or center_y + paddle_size[1]/2 >= window_dims[1] and self.direction > 0:
            self.direction *= -1

        return delta_y

class Speed_Limit:
    def __init__(self, speed_limit):
        # speed limit is given in height/sec
        self.speed_limit = speed_limit

    def decide(self, dt, ball_vel, ball_pos, window_dims, paddle_size, center_y):
        desired = ball_pos[1] - center_y
        if self.speed_limit == -1 or desired/dt <= self.speed_limit:
            return desired
        else:
            return self.speed_limit
