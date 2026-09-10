from abc import ABC, abstractmethod

class DeviceSettings:
    pass

class DeviceBackend(ABC):
    def __init__(self, settings=None):
        self._initial_settings = settings

    @abstractmethod
    def open(self, settings: DeviceSettings):
        raise NotImplementedError

    @abstractmethod
    def is_open(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def get_settings(self) -> DeviceSettings:
        return self._initial_settings

    def set_settings(self, settings: DeviceSettings=None) -> DeviceSettings:
        if settings:
            self._initial_settings = settings

        return self._initial_settings

    def get_source(self):
        raise NotImplementedError