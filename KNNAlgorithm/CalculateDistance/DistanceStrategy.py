from abc import abstractmethod, ABC

"""
 This abstract class defines the interface for all distance calculation strategies.
 The purpose is to provide a method to calculate the distances.
"""
class DistanceStrategy(ABC):
    @abstractmethod
    def calculate(self, point1, point2):
        pass