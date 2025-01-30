from abc import abstractmethod, ABC

from KNNAlgorithm.CalculateDistance.Strategy.SquaredEuclidianDistanceStrategy import SquaredEuclidianDistanceStrategy

"""
This is an abstract class designed to create distance calculation strategies.
It uses the Factory Method design pattern to ensure flexibility and scalability.    
The reason for using the Factory Method is to delegate the creation of different distance calculation strategies
to subclasses, allowing the system to work with various strategies.
"""
class DistanceCreator(ABC):
    @staticmethod
    def create_distance(strategy_type):
        if strategy_type == 1:
            return SquaredEuclidianDistanceStrategy()
        else:
            raise ValueError("Strategia non valida")

    def calculate_distance(self, strategy_type, point1, point2):

        strategy = self.create_distance(strategy_type)
        return strategy.calculate(point1, point2)