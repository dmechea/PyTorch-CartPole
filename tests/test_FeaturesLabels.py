import unittest
import sourceimporter
import torch
import numpy as np

FL = sourceimporter.FeaturesLabels

class testGameMemoryToFeatureSet(unittest.TestCase):
    def test_convertGameMemoryToArrays(self):

        exampleGameMemory = (
            (np.array([1,2,3,4]), [0.1, 0.2], 1, 0),
            (np.array([5,6,7,8]), [0.3, 0.4], 1, 0),
            (np.array([9,10,11,12]), [0.4, 0.5], 1, 0),
            (np.array([13,14,15,16]), [0.6, 0.7], 1, 0),
            (np.array([17,18,19,20]), [0.8, 0.9], 1, 0),
            (np.array([21,22,23,24]), [1.0, 1.1], 1, 0),
        )

        expectedFeatures = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
        [17,18,19,20],
        [21,22,23,24],
        ])

        expectedLabels = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.4, 0.5],
        [0.6, 0.7],
        [0.8, 0.9],
        [1.0, 1.1],
        ])

        resultFeatures, resultLabels = FL.memoryToFeatureLabel(
            exampleGameMemory,
        )

        self.assertTrue(np.all(resultFeatures == expectedFeatures))
        self.assertTrue(np.all(resultLabels == expectedLabels))


if __name__ == '__main__':
    unittest.main()
