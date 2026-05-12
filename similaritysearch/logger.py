from abc import ABC, abstractmethod
from typing_extensions import Self

class AbstractLogger (ABC):
    """
    Abstract Logger class contract for uniform logging across multiple log types
    """
    @abstractmethod
    def debug(self, msg: str):
        pass
    @abstractmethod
    def error(self, msg: str):
        pass
    def value(self, operation: str, val: str):
        pass

class Logger (AbstractLogger):
    """
    Concrete Singleton Logger
    """
    _instance = None
    
    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance


    def debug(self, msg: str):
        print(f"[DEBUG] {msg}")
    def error(self, msg: str):
        print(f"[ERROR] {msg}")
    def value(self, operation: str, val: str, **kwargs):
        args = ", ".join([f"{k}: {v}" for k,v in kwargs.items()])
        print(f"Value obtained for Operation {operation} with {args} was {val}")

logger = Logger()
