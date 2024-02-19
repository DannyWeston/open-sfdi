from fringe_app.controllers import MainController
from fringe_app.views import MainView

if __name__ == "__main__":
    mv = MainView()
    mc = MainController(model=None, view=mv)