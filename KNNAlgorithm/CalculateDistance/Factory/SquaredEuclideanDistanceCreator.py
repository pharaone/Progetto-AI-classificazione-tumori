from KNNAlgorithm.CalculateDistance.Strategy.DistanceStrategy import DistanceStrategy
from KNNAlgorithm.CalculateDistance.Strategy.EuclideanDistanceStrategy import EuclidianDistanceStrategy
from KNNAlgorithm.CalculateDistance.Factory.DistanceCreator import DistanceCreator
from KNNAlgorithm.CalculateDistance.Strategy.SquaredEuclidianDistanceStrategy import SquaredEuclidianDistanceStrategy

"""
This concrete class implements the DistanceCreator specifically for the Euclidean distance strategy without square root.
This ensures that adding new distance strategies does not require modifying this class.
"""
class SquaredEuclideanDistanceCreator(DistanceCreator):
    def create_distance(self)-> DistanceStrategy:
        return SquaredEuclidianDistanceStrategy()