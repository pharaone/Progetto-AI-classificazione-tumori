from KNNAlgorithm.CalculateDistance.DistanceStrategy import DistanceStrategy
from KNNAlgorithm.CalculateDistance.EuclideanDistanceStrategy import EuclidianDistanceStrategy
from KNNAlgorithm.CalculateDistance.Factory.DistanceCreator import DistanceCreator


class EuclideanDistanceCreator(DistanceCreator):
    def create_distance(self)-> DistanceStrategy:
        return EuclidianDistanceStrategy()