from abc import ABC, abstractmethod


class Architecture(ABC):
    """Abstract class for distributed architectures"""

    @abstractmethod
    def _spawn(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
