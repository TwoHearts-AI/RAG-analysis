from abc import ABC, abstractmethod

class RetrieverABC(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def search_relevant_context(self, query: str):
        pass