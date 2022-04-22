import json
import os
import warnings
from typing import Dict, List

from model import GameModel
from player import Player, Human, Dummy, NeuralNet, Classic
from view import GUIController

TRAIN_GAMES = 10000
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "TRAIN_MODE": False,
    "positions": {
        "player1": [0.1, 0.5],
        "player2": [0.9, 0.5],
        "ball": [0.5, 0.5],
        "ball_vel": [0.5, 0.0]
    },
    "players": ["Human", "Classic"],
    "resolution": [1200, 600],
    "game_speed": 1.0,
    "limits": {
        "max_speed_ball": 10000,
        "speedup": 1.0,
        "points_to_win": 11,
        "max_speed_player": 0.02,
        "path_resolution": 100
    }
}

_train_game_runs = True
_last_game_won_by = -1
_target_update_after_steps = 10


def load_config_or_write_defaults() -> Dict:
    """
    loads config from config.json, or writes it if it doesn't exist
    :return: config dictionary
    """
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as fp:
            cfg = json.load(fp)
        
        cfg["positions"]["ball_vel"][0] *= cfg["game_speed"]
        cfg["positions"]["ball_vel"][1] *= cfg["game_speed"]
        cfg["limits"]["max_speed_ball"] *= cfg["game_speed"]
        cfg["limits"]["max_speed_player"] *= cfg["game_speed"]
        return cfg
    else:
        with open(CONFIG_FILE, "w+") as fp:
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
        elif player == "NeuralNet":
            args = {
                "id": id,
                "speed_limit": cfg["limits"]["max_speed_player"],
                "training": cfg["TRAIN_MODE"]
            }
            if cfg["TRAIN_MODE"]:
                args["epsilon"] = 0
            players.append(
                NeuralNet(**args)
            )
        elif player == "Classic":
            players.append(Classic(id))
        else:
            raise Exception("AI Type not understood!")

    return players


def training_game_over_callback(pid: int):
    global _train_game_runs, _last_game_won_by
    _train_game_runs = False
    _last_game_won_by = pid


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
        # get the NeuralNet players
        ais = []
        if isinstance(player1, NeuralNet):
            ais.append(player1)
        if isinstance(player2, NeuralNet):
            ais.append(player2)
        if not ais:
            warnings.warn("No trainable Ais detected! Running GUI-less without training is only "
                          "recommended for debugging")

        model.won_callback = training_game_over_callback
        for e in range(TRAIN_GAMES):
            print(f"game {e+1} of {TRAIN_GAMES}")
            # simulate the game
            gui = GUIController(cfg, model.human_input, model.update, model.reset, headless=True)
            model.gui_update = gui.update
            model.won_callback = gui.game_won
            gui.run()

            ais[0].train()
            # only the first will be trained to save time
            if len(ais) > 1:
                ais[1].reload_from_path(ais[0].model_path)

            if e % _target_update_after_steps == 0:
                print("updating target model...")
                ais[0].update_target_model()

            _train_game_runs = True
            model.reset()
