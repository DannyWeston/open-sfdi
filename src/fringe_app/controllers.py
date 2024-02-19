from PIL import Image, ImageTk
from sfdi.fringes import Fringes

import numpy as np

class MainController:
    def __init__(self, model, view) -> None:
        self.view = view
        self.model = model

        self.fringes = None

        self.showing = 0

        # These commands must be ran last
        self.view.add_callback('button', self.on_button_pressed)    # Callbacks

        self.view.bind_commands()                                   # Tell view to register callbacks 
        self.view.run()                                             # Present the view

    def generate_new_fringes(self):
        self.fringes = Fringes.from_generator(200, 200, 32).to_rgb()

    def on_button_pressed(self):
        if self.fringes is None:
            self.generate_new_fringes()
            self.showing = 0

        print(len(self.fringes))

        pil_img = Image.fromarray(self.fringes[self.showing]).convert('RGB')
        img = ImageTk.PhotoImage(pil_img)

        self.view.show_image(img)

