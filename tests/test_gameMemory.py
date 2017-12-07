import unittest
from sourceimporter import gameMemory

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
        result = gameMemory.addToGameMemory(maxMemorySize, eventToAdd, currentGameMemory)
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
        result = gameMemory.addToGameMemory(maxMemorySize, eventToAdd, currentGameMemory)
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



if __name__ == '__main__':
    unittest.main()
