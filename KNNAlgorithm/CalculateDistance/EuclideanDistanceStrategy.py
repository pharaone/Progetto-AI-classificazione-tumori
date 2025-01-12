import numpy as np

from KNNAlgorithm.CalculateDistance.DistanceStrategy import DistanceStrategy

class EuclidianDistanceStrategy(DistanceStrategy):

    def __init__(self):
        pass


    def calculate(self, point1, point2):
        point3 = point1 - point2                                # Difference between the two points
        distances = np.sqrt(np.sum(pow(point3, 2)))             # Calculation of the Euclidean distance
        return distances