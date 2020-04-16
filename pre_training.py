from tensorflow import keras as K
import numpy as np

"""
pre-trains the neural net to follow the ball with supervised learning
"""

model_path = "models/DeepQPong"
batch_size = 256
tol = 1.0/12.0

def create_model():
    # all features need to be normalized respective to the player
    # features: dt, my_y, enemy_y, ball_x, ball_y, ball_vel_x, ball_vel_y
    inps = K.layers.Input(shape=(7,))
    # bottleneck layer, maybe it learns how to remove useless info
    x = K.layers.Dense(6, activation="selu")(inps)
    # the actual hidden layers
    x = K.layers.Dense(128, activation="selu")(x)
    x = K.layers.Dense(256, activation="selu")(x)
    x = K.layers.Dense(64, activation="selu")(x)
    # output
    x = K.layers.Dense(3, activation="linear")(x)
    model = K.Model(inputs=inps, outputs=x)
    model.compile(loss="MSE", optimizer="adam")
    return model

class DataGenerator(K.utils.Sequence):
    def __getitem__(self, item):
        while True:
            X = np.zeros((batch_size, 9))
            # fill dt
            X[:, 0] = 0.01 + (np.random.rand(batch_size) - 0.5) * 0.01
            # fill my_y
            X[:, 1] = np.random.rand(batch_size)
            # enemy_y
            X[:, 2] = np.random.rand(batch_size)
            # ball_x
            X[:, 3] = np.random.rand(batch_size)
            # ball_y
            X[:, 4] = np.random.rand(batch_size)
            # ball_vel_x
            X[:, 5] = np.random.rand(batch_size)
            # ball_vel_y
            X[:, 6] = np.random.rand(batch_size)

            y = np.array([[0, 0, 1] if x[4] - x[1] < - tol else ([1, 0, 0] if x[4] - x[1] > tol else [0, 1, 0]) for x in X])

            yield X, y


model = create_model()
model.fit_generator(generator=DataGenerator(), validation_data=DataGenerator(), use_multiprocessing=True, workers=8)
