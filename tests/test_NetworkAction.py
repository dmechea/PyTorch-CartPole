import unittest
import sourceimporter
import torch
import numpy as np
from torch.autograd import Variable
ActionSpace = sourceimporter.ActionSpace
NN = sourceimporter.Networks


class testActionSpaceFuncs(unittest.TestCase):
    def test_explorationDecay(self):
        initialExpl = 0.8
        finalExpl = 0.2
        episodeNumbers = 4

        result = ActionSpace.explorationDecay(
                                initialExpl,
                                finalExpl,
                                episodeNumbers,
        )
        expected = 0.15

        self.assertEqual(int(result*100), int(expected*100))

    def test_decayAfterTimeStep(self):
        initialExpl = 0.8
        finalExpl = 0.2
        episodeNumbers = 4
        stepSize = ActionSpace.explorationDecay(
            initialExpl,
            finalExpl,
            episodeNumbers,
        )
        expected1 = 0.65
        expected2 = 0.5
        result = ActionSpace.updatedExploration(initialExpl, finalExpl, stepSize)
        self.assertEqual(int(result*100), int(expected1*100))
        result = ActionSpace.updatedExploration(result, finalExpl, stepSize)
        self.assertEqual(int(result*100), int(expected2*100))

    def test_convertNumpyStateToVariable(self):
        dummyState = np.array([0.5, 0.04, -0.4324324, 4.3])
        result = ActionSpace.convertToVariable(
            dummyState,
            torch.FloatTensor,
        )
        self.assertEqual(type(result.data), torch.FloatTensor)

    def test_outputAndPredictionSizing(self):
        TestNet = NN.LinearNet(4, 10, 2)
        dummyState1 = np.array([0.5, 0.04, -0.4324324, 4.3])
        prediction = ActionSpace.makePrediction(TestNet, dummyState1)
        predictionSize = prediction.size()
        expectedSize = torch.Size([2])

        self.assertEqual(predictionSize, expectedSize)

        dummyState2 = np.array([
            [0.5, 0.04, -0.4324324, 4.3],
            [-0.5, 0.34, -0.4324324, 9.3],
            [-0.67, 2.34, 0.6324324, 3.3],
        ])

        prediction2 = ActionSpace.makePrediction(TestNet, dummyState2)
        prediction2Size = prediction2.size()
        expectedSize = torch.Size([3, 2])

        self.assertEqual(len(prediction2[0]), 2)
        self.assertEqual(prediction2Size, expectedSize)

    def test_randomIsInRange(self):
        bottomLimit = 0
        upperLimit = 1

        sample = ActionSpace.getRandomSample()
        self.assertGreaterEqual(sample, bottomLimit)
        self.assertLessEqual(sample, upperLimit)

    def test_selectSomeActions(self):
        TestNet = NN.LinearTwoDeep(4, 10, 15, 2)
        CurrentExpl, finalExpl, epNums = (0.5, 0.2, 10)
        explDecay = ActionSpace.explorationDecay(CurrentExpl, finalExpl, epNums)
        dummyState = np.array([0.5, 0.04, -0.4324324, 4.3])
        prediction = ActionSpace.makePrediction(TestNet, dummyState)
        sample = ActionSpace.getRandomSample()

        action = ActionSpace.selectAction(
            2,
            prediction,
            CurrentExpl,
        )
        possibleChoices = (0, 1)
        self.assertIn(action, possibleChoices)

if __name__ == '__main__':
    unittest.main()
