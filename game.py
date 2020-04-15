import json
import os
from sys import argv
from typing import Dict, List

from model import GameModel
from player import Player, Human, Dummy, NeuralNet, Heuristic
from view import GUIController

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "TRAIN_MODE": False,
    "positions": {
        "player1": [0.1, 0.5],
        "player2": [0.9, 0.5],
        "ball": [0.5, 0.5],
        "ball_vel": [0.5, 0.0]
    },
    "players": ["Human", "Heuristic"],
    "limits": {
        "resolution": [800, 600],
        "max_speed_ball": 10000,
        "speedup": 1.05,
        "points_to_win": 11,
        "max_speed_player": 0.02,
        "path_resolution": 100
    }
}


def load_config_or_write_defaults() -> Dict:
    """
    loads config from config.json, or writes it if it doesn't exist
    :return: config dictionary
    """
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as fp:
            cfg = json.load(fp)
        return cfg
    else:
        with open(CONFIG_FILE, "w+") as fp:
            DEFAULT_CONFIG["TRAIN_MODE"] = "--nogui" in argv
            json.dump(DEFAULT_CONFIG, fp, indent=2)
        return DEFAULT_CONFIG


def get_players(cfg: Dict, player1_args: Dict = None, player2_args: Dict = None) -> List[Player]:
    """
    builds a list of Player-objects from the game config
    :param cfg: Game config
    :param player1_args: arguments for the constructor of player1
    :param player2_args: arguments for the constructor of player 2
    :return: [Player1, Player2]
    """
    if player2_args is None:
        player2_args = {}
    if player1_args is None:
        player1_args = {}
    players: List[Player] = []
    args_ls = [player1_args, player2_args]
    for id, (player, args) in enumerate(zip(cfg["players"], args_ls)):
        if player == "Human":
            players.append(Human(id))
        elif player == "Dummy":
            players.append(Dummy(id))
        elif player == "Heuristic":
            players.append(Heuristic(id))
        elif player == "NeuralNet":
            players.append(NeuralNet(id))
        else:
            raise Exception("AI Type not understood!")

    return players


if __name__ == "__main__":
    cfg = load_config_or_write_defaults()
    player1, player2 = get_players(cfg)
    model = GameModel(cfg, player1, player2)
    if not cfg["TRAIN_MODE"]:
        # run with GUI, no training
        gui = GUIController(cfg, model.human_input, model.update, model.reset)
        model.gui_update = gui.update
        model.won_callback = gui.game_won
        gui.run()

    else:
        # TODO: run without GUI and train
        pass

