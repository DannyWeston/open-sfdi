
from sfdi.fringes import Fringes
from fringe_app.views import MainView
from fringe_app.models import FringesModel

from sfdi.io.repositories import ImageRepository

import numpy as np

class MainController:
    def __init__(self, model = FringesModel()) -> None:
        self.model = model

        self.showing = 0

        # These commands must be ran last
        self.view = MainView(self.model)
        
        self.view.add_callback('next_button', self.on_next_button_pressed)
        self.view.add_callback('save_button', self.on_save_button_pressed)
        
        self.view.add_callback('n_slider', self.on_slider_update)
        self.view.add_callback('freq_slider', self.on_slider_update)
        self.view.add_callback('size_slider', self.on_slider_update)
        self.view.add_callback('angle_slider', self.on_slider_update)

        self.view.bind_commands() # Tell view to register callbacks
        self.view.run() # Present the view
    
    def on_save_button_pressed(self, *values):
        import tkinter.filedialog as fd
        folder = fd.askdirectory()
        
        if folder is None or folder == '': return
        
        image_repo = ImageRepository(folder)
        
        width, height, freq, orientation, n, fringe_type = values
        
        fringes = Fringes.from_generator(width, height, freq, orientation, n, fringe_type)
        
        for i, img in enumerate(fringes):
            image_repo.save(img, name=f'fringes{i}')
    
    def on_slider_update(self, *args):
        width, height, freq, orientation, n, fringe_type = args
        
        self.model.update_fringes(width, height, freq, orientation, n, fringe_type)

    def on_next_button_pressed(self):
        self.model.next_fringes()