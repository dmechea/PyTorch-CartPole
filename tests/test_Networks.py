import unittest
import sourceimporter
import torch

NN = sourceimporter.Networks

class testNetworkStructures(unittest.TestCase):

    def test_createSimpleNetwork(self):
        TestNet = NN.LinearNet(4, 10, 2)

        NetworkLayers = len(TestNet.getParameters())
        LayerExpectation = 4
        self.assertEqual(NetworkLayers, LayerExpectation)

        firstMatrixSize = TestNet.getParameters()[0].size()
        expectedFirstSize = torch.Size([10,4])
        self.assertEqual(firstMatrixSize, expectedFirstSize)

        secondMatrixSize = TestNet.getParameters()[1].size()
        expectedSecondSize = torch.Size([10])
        self.assertEqual(secondMatrixSize, expectedSecondSize)

        thirdMatrixSize = TestNet.getParameters()[2].size()
        expectedThirdSize = torch.Size([2, 10])
        self.assertEqual(thirdMatrixSize, expectedThirdSize)

        fourthMatrixSize = TestNet.getParameters()[3].size()
        expectedFourthSize = torch.Size([2])
        self.assertEqual(fourthMatrixSize, expectedFourthSize)


    def test_checkTwoDeepStructure(self):
        TestNet = NN.LinearTwoDeep(4, 10, 15, 2)

        NetworkLayers = len(TestNet.getParameters())
        LayerExpectation = 6
        self.assertEqual(NetworkLayers, LayerExpectation)

        firstMatrixSize = TestNet.getParameters()[0].size()
        expectedFirstSize = torch.Size([10,4])
        self.assertEqual(firstMatrixSize, expectedFirstSize)

        secondMatrixSize = TestNet.getParameters()[1].size()
        expectedSecondSize = torch.Size([10])
        self.assertEqual(secondMatrixSize, expectedSecondSize)

        thirdMatrixSize = TestNet.getParameters()[2].size()
        expectedThirdSize = torch.Size([15, 10])
        self.assertEqual(thirdMatrixSize, expectedThirdSize)

        fourthMatrixSize = TestNet.getParameters()[3].size()
        expectedFourthSize = torch.Size([15])
        self.assertEqual(fourthMatrixSize, expectedFourthSize)

        fifthMatrixSize = TestNet.getParameters()[4].size()
        expectedFifthSize = torch.Size([2, 15])
        self.assertEqual(fifthMatrixSize, expectedFifthSize)

        sixthMatrixSize = TestNet.getParameters()[5].size()
        expectedSixthSize = torch.Size([2])
        self.assertEqual(sixthMatrixSize, expectedSixthSize)


    def test_checkThreeDeepStructure(self):
        TestNet = NN.LinearThreeDeep(4, 128, 256, 128, 2)

        NetworkLayers = len(TestNet.getParameters())
        LayerExpectation = 8
        self.assertEqual(NetworkLayers, LayerExpectation)

        firstMatrixSize = TestNet.getParameters()[0].size()
        expectedFirstSize = torch.Size([128,4])
        self.assertEqual(firstMatrixSize, expectedFirstSize)

        secondMatrixSize = TestNet.getParameters()[1].size()
        expectedSecondSize = torch.Size([128])
        self.assertEqual(secondMatrixSize, expectedSecondSize)

        thirdMatrixSize = TestNet.getParameters()[2].size()
        expectedThirdSize = torch.Size([256, 128])
        self.assertEqual(thirdMatrixSize, expectedThirdSize)

        fourthMatrixSize = TestNet.getParameters()[3].size()
        expectedFourthSize = torch.Size([256])
        self.assertEqual(fourthMatrixSize, expectedFourthSize)

        fifthMatrixSize = TestNet.getParameters()[4].size()
        expectedFifthSize = torch.Size([128, 256])
        self.assertEqual(fifthMatrixSize, expectedFifthSize)

        sixthMatrixSize = TestNet.getParameters()[5].size()
        expectedSixthSize = torch.Size([128])
        self.assertEqual(sixthMatrixSize, expectedSixthSize)

        seventhMatrixSize = TestNet.getParameters()[6].size()
        expectedSeventhSize = torch.Size([2, 128])
        self.assertEqual(seventhMatrixSize, expectedSeventhSize)

        eighthMatrixSize = TestNet.getParameters()[7].size()
        expectedEighthSize = torch.Size([2])
        self.assertEqual(eighthMatrixSize, expectedEighthSize)

if __name__ == '__main__':
    unittest.main()
