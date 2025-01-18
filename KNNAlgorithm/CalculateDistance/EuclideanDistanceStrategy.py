import numpy as np

from KNNAlgorithm.CalculateDistance.DistanceStrategy import DistanceStrategy
"""
This concrete class implements the DistanceStrategy for calculating the Euclidean distance between two points.
By implementing this as a strategy, the system can easily switch to other distance calculations when needed.
"""
class EuclidianDistanceStrategy(DistanceStrategy):

    def calculate(self, point1, point2):
        point3 = point1 - point2                                # Difference between the two points
        distances = np.sqrt(np.sum(pow(point3, 2)))             # Calculation of the Euclidean distance
        return distances