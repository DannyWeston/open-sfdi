import tkinter
from tkinter import ttk
import sv_ttk

class View():
    def __init__(self, title, window_size):
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

class MainView(View):
    def __init__(self, title='Fringe Projection App', window_size=(800, 600)):
        super().__init__(title, window_size)

        self.img_canvas = tkinter.Canvas(self.root, width=window_size[0], height=window_size[1] - 100)
        self.img_canvas.pack()

        self.button = ttk.Button(self.root, text="Button")
        self.button.pack()

        self.img_on_canvas = self.img_canvas.create_image(0, 0, anchor='nw', image=None)

    def show_image(self, img):
        self.img_canvas.itemconfig(self.img_on_canvas, image=img)


    def bind_commands(self):
        self.button.config(command=self.callbacks['button'])