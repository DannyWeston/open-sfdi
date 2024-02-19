import tkinter
from tkinter import ttk
import sv_ttk
from PIL import Image, ImageTk
import numpy as np

from abc import ABC, abstractmethod

from fringe_app.models import FringesModel
    
class Observer(ABC):
    @abstractmethod
    def on_update(self, value, *args):
        raise NotImplementedError

class View():
    def __init__(self, model, title, window_size):
        self.model = model
        
        self.root = tkinter.Tk()

        self.callbacks = {}

        sv_ttk.use_dark_theme()

        self.root.title(title)

        self.root.geometry(f"{window_size[0]}x{window_size[1]}")

    def set_window_size(self, window_size):
        self.root.geometry(f"{window_size[0]}x{window_size[1]}")

    def run(self):
        self.root.mainloop()

    def add_callback(self, name, cb):
        self.callbacks[name] = cb

    def bind_commands(self):
        raise NotImplementedError

class MainView(View, Observer):
    def __init__(self, model: FringesModel, title='Fringe Projection App', window_size=(800, 800)):
        super().__init__(model, title, window_size)
        
        self.image_width = self.image_height = int(window_size[1] / 2)
        
        self.root.grid_columnconfigure((0, 1), weight=1)

        self.canvas = tkinter.Canvas(self.root, width=self.image_width, height=self.image_height)
        self.canvas.grid(row=0, column=0, columnspan=2, pady=(0, 100))
        
        
        self.label_n_slider = ttk.Label(self.root, text="Phase Count: -", width=20)
        self.label_n_slider.grid(row=1, column=0, pady=(0, 20))
        
        self.n_slider = ttk.Scale(self.root, from_=3, to=16)
        self.n_slider.set(3)
        self.n_slider.grid(row=1, column=1, pady=(0, 20))
        

        self.label_freq_slider = ttk.Label(self.root, text="SF (Pixels): -", width=20)
        self.label_freq_slider.grid(row=2, column=0, pady=(0, 20))
        
        self.freq_slider = ttk.Scale(self.root, from_=4, to=128)
        self.freq_slider.set(32)
        self.freq_slider.grid(row=2, column=1, pady=(0, 20))
        
        
        self.label_size_slider = ttk.Label(self.root, text="Image Size (Pixels): -", width=20)
        self.label_size_slider.grid(row=3, column=0, pady=(0, 20))
        
        self.size_slider = ttk.Scale(self.root, from_=128, to=1024)
        self.size_slider.set(512)
        self.size_slider.grid(row=3, column=1, pady=(0, 20))
        
        
        self.label_angle_slider = ttk.Label(self.root, text="Orientation:", width=20)
        self.label_angle_slider.grid(row=4, column=0, pady=(0, 20))
        
        self.angle_slider = ttk.Scale(self.root, from_=0.0, to=1.0)
        self.angle_slider.set(0)
        self.angle_slider.grid(row=4, column=1, pady=(0, 20))
        
        self.next_button = ttk.Button(self.root, text="Next")
        self.next_button.grid(row=5, column=0)
        
        self.save_button = ttk.Button(self.root, text="Save")
        self.save_button.grid(row=5, column=1)
        
        self.GARBAGE_BUG_IMG = self.model.get_fringes() # https://stackoverflow.com/questions/16424091/why-does-tkinter-image-not-show-up-if-created-in-a-function
        
        self.img_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=self.convert_image(self.GARBAGE_BUG_IMG))
        
        self.model.add_listener(self)

    def on_update(self, img, *args):
        self.canvas.image = img
        self.canvas.itemconfig(self.img_on_canvas, image=self.convert_image(img)) 
        
        if args is None or len(args) <= 0: return
        
        width, height, freq, orientation, n, fringe_type = args
        
        if width: self.label_size_slider.config(text=f"Image Size (Pixels): {width}")
        elif height: self.label_size_slider.config(text=f"Image Size (Pixels): {height}")
        
        if orientation: self.label_angle_slider.config(text=f"Orientation:")
            
        if freq: self.label_freq_slider.config(text=f"SF (Pixels): {freq}")
        if n: self.label_n_slider.config(text=f"Phase Count: {n}")

    def convert_image(self, img):
        if img is None: return None
        
        pil_img = Image.fromarray(img).convert('RGB')
        crop_img = pil_img.resize((self.image_width, self.image_height), Image.LANCZOS) 
        
        temp = ImageTk.PhotoImage(crop_img)
        self.GARBAGE_BUG_IMG = temp

        return temp

    def get_values(self):
        n = int(float(self.n_slider.get()))
        freq = int(float(self.freq_slider.get()))
        width = height = int(float(self.size_slider.get()))
        orientation = float(self.angle_slider.get()) * (np.pi / 2.0)
        
        return width, height, freq, orientation, n, 'Sinusoidal'

    def bind_commands(self):
        self.next_button.config(command=self.callbacks['next_button'])
        self.save_button.config(command=lambda: self.callbacks['save_button'](*self.get_values()))
        
        self.n_slider.config(command=lambda _: self.callbacks['n_slider'](*self.get_values()))
        self.freq_slider.config(command=lambda _: self.callbacks['freq_slider'](*self.get_values()))
        self.size_slider.config(command=lambda _: self.callbacks['size_slider'](*self.get_values()))
        self.angle_slider.config(command=self.zero_or_one)
        
    def zero_or_one(self, something):
        value = self.angle_slider.get()
        if value != int(value): 
            self.angle_slider.set(round(value))
            
        self.callbacks['angle_slider'](*self.get_values())