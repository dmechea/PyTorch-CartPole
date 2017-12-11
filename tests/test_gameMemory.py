import unittest
import sourceimporter
import numpy as np
import torch

gameMemory = sourceimporter.gameMemory
NN = sourceimporter.Networks
act = sourceimporter.ActionSpace

class TestgameMemoryFuncs(unittest.TestCase):
    def test_addToGameMemory(self):
        maxMemorySize = 5
        eventToAdd = (35, 3, 40, 1)
        currentGameMemory = (
            (30, 1, 35, 1),
            (25, 1, 30, 1),
            )
        expected = (
            (35, 3, 40, 1),
            (30, 1, 35, 1),
            (25, 1, 30, 1),
            )
        maxxedOut = gameMemory.isFull(maxMemorySize, currentGameMemory)
        result = gameMemory.addToGameMemory(
            maxxedOut,
            eventToAdd,
            currentGameMemory,
        )
        self.assertEqual(result, expected)

    def test_addToFullGameMemory(self):
        maxMemorySize = 2
        eventToAdd = (35, 3, 40, 1)
        currentGameMemory = (
            (30, 1, 35, 1),
            (25, 1, 30, 1),
            )
        expected = (
            (35, 3, 40, 1),
            (30, 1, 35, 1),
            )
        maxxedOut = gameMemory.isFull(maxMemorySize, currentGameMemory)
        result = gameMemory.addToGameMemory(
            maxxedOut,
            eventToAdd,
            currentGameMemory
        )
        self.assertEqual(result, expected)

    def test_batchSample1(self):
        currentGameMemory = (
            (35, 3, 40, 1),
            (30, 1, 35, 1),
            (25, 1, 30, 1),
        )
        batchSize = 1
        result = gameMemory.batchSample(batchSize, currentGameMemory)
        self.assertEqual(len(result), batchSize)

    def test_batchSample2(self):
        currentGameMemory = (
            (35, 3, 40, 1),
            (30, 1, 35, 1),
            (25, 1, 30, 1),
        )
        batchSize = 2
        result = gameMemory.batchSample(batchSize, currentGameMemory)
        self.assertIn(result[0], currentGameMemory)
        self.assertIn(result[1], currentGameMemory)

    def test_addToEmptySequence(self):
        dummyGameMemory = ( 1, 2, 3, 4 )
        expected = (
            (1, 2, 3, 4),
        )
        result = gameMemory.addToSequence(dummyGameMemory)
        self.assertEqual(result, expected)

    def test_addToSingularSequence(self):
        dummyGameMemory = ( 1, 2, 3, 4 )
        currentSequence = (
            (2, 3, 4, 5),
        )
        expected = (
            (1, 2, 3, 4),
            (2, 3, 4, 5),
        )
        result = gameMemory.addToSequence(dummyGameMemory, currentSequence)
        self.assertEqual(result, expected)

    def test_addToMultipleSequence(self):
        dummyGameMemory = ( 1, 2, 3, 4 )
        currentSequence = (
            (2, 3, 4, 5),
            (3, 4, 5, 6),
        )
        expected = (
            (1, 2, 3, 4),
            (2, 3, 4, 5),
            (3, 4, 5, 6),
        )
        result = gameMemory.addToSequence(dummyGameMemory, currentSequence)
        self.assertEqual(result, expected)

    def test_QValueSequenceUpdate(self):
        learningRate = 0.5
        discount = 0.9
        state1 = np.array([1, 2, 3, 4])
        state2 = np.array([2, 3, 4, 5])
        state3 = np.array([3, 4, 5, 6])
        InitQValues1 = [0, 0]
        InitQValues2 = [0, 0]
        InitQValues3 = [0, 0]
        ExpectedQ1 = [0, 0]
        ExpectedQ2 = [0, 0.5]
        ExpectedQ3 = [0.725, 0]
        Action1 = 0
        Action2 = 1
        Action3 = 0
        Reward1 = 0
        Reward2 = 1
        Reward3 = 1

        mem3 = gameMemory.preProcessedMemory(
            state3, InitQValues3, Action3, Reward3
        )
        mem2 = gameMemory.preProcessedMemory(
            state2, InitQValues2, Action2, Reward2
        )
        mem1 = gameMemory.preProcessedMemory(
            state1, InitQValues1, Action1, Reward1
        )
        currentSeqMem = gameMemory.addToSequence(mem3)
        currentSeqMem = gameMemory.addToSequence(mem2, currentSeqMem)
        currentSeqMem = gameMemory.addToSequence(mem1, currentSeqMem)

        result = gameMemory.modifyQValues(
            currentSeqMem,
            learningRate,
            discount,
            gameMemory.addToSequence
        )

        # Expected Result
        mem3 = gameMemory.preProcessedMemory(
            state3, ExpectedQ3, Action3, Reward3
        )
        mem2 = gameMemory.preProcessedMemory(
            state2, ExpectedQ2, Action2, Reward2
        )
        mem1 = gameMemory.preProcessedMemory(
            state1, ExpectedQ1, Action1, Reward1
        )

        ExpectedSeqMem = gameMemory.addToSequence(mem1)
        ExpectedSeqMem = gameMemory.addToSequence(mem2, ExpectedSeqMem)
        ExpectedSeqMem = gameMemory.addToSequence(mem3, ExpectedSeqMem)

        self.assertEqual(result, ExpectedSeqMem)

if __name__ == '__main__':
    unittest.main()
