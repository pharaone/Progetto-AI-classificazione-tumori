from KNNAlgorithm.CalculateDistance.DistanceStrategy import DistanceStrategy
from KNNAlgorithm.CalculateDistance.EuclideanDistanceStrategy import EuclidianDistanceStrategy
from KNNAlgorithm.CalculateDistance.Factory.DistanceCreator import DistanceCreator

"""
This concrete class implements the DistanceCreator specifically for the Euclidean distance strategy.
This ensures that adding new distance strategies does not require modifying this class.
"""
class EuclideanDistanceCreator(DistanceCreator):
    def create_distance(self)-> DistanceStrategy:
        return EuclidianDistanceStrategy()