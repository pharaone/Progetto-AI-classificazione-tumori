from abc import abstractmethod, ABC


class DistanceStrategy(ABC):
    @abstractmethod
    def calculate(self, point1, point2):
        pass