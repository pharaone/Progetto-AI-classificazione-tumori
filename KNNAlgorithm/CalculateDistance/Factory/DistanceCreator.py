from abc import abstractmethod, ABC

from KNNAlgorithm.CalculateDistance.Strategy.DistanceStrategy import DistanceStrategy
"""
This is an abstract class designed to create distance calculation strategies.
It uses the Factory Method design pattern to ensure flexibility and scalability.    
The reason for using the Factory Method is to delegate the creation of different distance calculation strategies
to subclasses, allowing the system to work with various strategies.
"""
class DistanceCreator(ABC):
    @abstractmethod
    def create_distance(self) -> DistanceStrategy:
        pass

    def calculate_distance(self, point1, point2):

        strategy = self.create_distance()
        return strategy.calculate(point1, point2)