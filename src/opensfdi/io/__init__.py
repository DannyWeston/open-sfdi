
from abc import ABC, abstractmethod


class IRepository(ABC):
    # @abstractmethod
    # def commit(self):
    #     pass

    @abstractmethod
    def get(self, id:int):
        pass

    @abstractmethod
    def add(self, **kwargs) -> int:
        pass

    @abstractmethod
    def delete(self, id:int):
        pass

    @abstractmethod
    def update(self, id:int, **kwargs):
        pass