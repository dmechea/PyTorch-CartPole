import unittest
import sourceimporter
import torch
import numpy as np

Cuda = sourceimporter.Cuda
NN = sourceimporter.Networks
act = sourceimporter.ActionSpace

class testPerfomGPUCalc(unittest.TestCase):
    def test_NetworkGPUPrediction(self):
        CPUTestNet = NN.LinearTwoDeep(4, 10, 10, 2)
        GPUTestNet = Cuda.assignToGPU(CPUTestNet)

        OtherTestNet = NN.LinearTwoDeep(4, 5, 5, 2)
        CPUdataType = torch.FloatTensor
        GPUdataType = torch.cuda.FloatTensor

        dummyState = np.array([
            0.3543728532,
            0.50987907,
            -0.6654636543,
            0.32713878,
        ])

        CPUInFeed = act.convertToVariable(dummyState, CPUdataType)
        GPUInFeed = act.convertToVariable(dummyState, GPUdataType)

        with self.assertRaises(TypeError):
            act.makePrediction(CPUTestNet, GPUInFeed)
        with self.assertRaises(TypeError):
            act.makePrediction(GPUTestNet, CPUInFeed)

        shouldRunOnCPU = act.makePrediction(CPUTestNet, CPUInFeed)
        shouldRunOnGPU = act.makePrediction(GPUTestNet, GPUInFeed)
        self.assertEqual(type(shouldRunOnCPU.data), CPUdataType)
        self.assertEqual(type(shouldRunOnGPU.data), GPUdataType)

if __name__ == '__main__':
    unittest.main()
