from abc import abstractmethod, ABC

from KNNAlgorithm.CalculateDistance.DistanceStrategy import DistanceStrategy


class DistanceCreator(ABC):
    @abstractmethod
    def create_distance(self) -> DistanceStrategy:
        pass

    def calculate_distance(self, point1, point2):

        strategy = self.create_distance()
        return strategy.calculate(point1, point2)