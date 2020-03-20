from model import GameModel
from view import GUIController

if __name__ == "__main__":
    model = GameModel()
    gui = GUIController(*model.get_initial_positions())
    gui.run()
