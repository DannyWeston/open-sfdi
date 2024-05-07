from abc import ABC, abstractmethod

from sfdi.fringes import FringeGenerator


import numpy as np


class Observable(ABC):

    def notify(self, value, *args):

        for listener in self.listeners:

            listener.on_update(value, *args)
    

    @abstractmethod

    def add_listener(self, listener):

        raise NotImplementedError


class FringesModel(Observable):

    def __init__(self, fringes=None, listeners=[]):

        super().__init__()
        

        self.listeners = listeners
        

        self.fringes = fringes

        self.viewing = 0
    

    def get_fringes(self):

        if self.fringes is None or len(self.fringes) <= 0: return None
        

        return self.fringes[self.viewing]


    def next_fringes(self):

        if self.fringes is None or len(self.fringes) == 0: return None
        

        self.viewing = (self.viewing + 1) % len(self.fringes)
        

        temp = self.fringes[self.viewing]
        

        self.notify(temp) # Notify any listeners
        

        return temp


    def update_fringes(self, width, height, freq, orientation, n, fringe_type='Sinusoidal'):

        fringes = FringeGenerator.from_generator(width, height, freq, orientation, n, fringe_type)
        

        self.fringes = fringes

        self.viewing = 0

        self.notify(self.get_fringes(), width, height, freq, orientation, n, fringe_type)


    def add_listener(self, listener):
        self.listeners.append(listener)