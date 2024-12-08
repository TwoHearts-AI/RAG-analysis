from abc import ABC, abstractmethod

class GeneratorABC(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, query):
        pass