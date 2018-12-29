import os
import json

class Config():
    cfgf = './config.cfg'
    cfg = {
        'frame_limit': 300,
        'players': ['Human', ['Heuristic', -1]],
        'points_to_win': 11,
        'speedup': 1.1,
        'offset': 0.02,
    }
    @staticmethod
    def load():
        if not os.path.isfile(Config.cfgf):
            Config.write_config()
        else:
            with open(Config.cfgf, 'r') as f:
                custom = json.loads(f.read())
                for key in custom:
                    Config.cfg[key] = custom[key]
    @staticmethod
    def write_config():
        with open(Config.cfgf, 'w+') as f:
            f.write(json.dumps(Config.cfg, sort_keys=True, indent=4))

    @staticmethod
    def get(key):
        return Config.cfg[key]

    @staticmethod
    def set(key, value):
        Config.cfg[key] = value
        Config.write_config()
