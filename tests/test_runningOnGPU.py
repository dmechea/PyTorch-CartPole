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

        dummyState = np.array([
            0.3543728532,
            0.50987907,
            -0.6654636543,
            0.32713878,
        ])

        with self.assertRaises(TypeError):
            act.makePrediction(CPUTestNet, dummyState, isGPU = True)
        with self.assertRaises(TypeError):
            act.makePrediction(GPUTestNet, dummyState)

        shouldRunOnCPU = act.makePrediction(CPUTestNet, dummyState)
        shouldRunOnGPU = act.makePrediction(
            GPUTestNet,
            dummyState,
            isGPU = True,
        )
        self.assertEqual(type(shouldRunOnCPU.data), torch.FloatTensor)
        self.assertEqual(type(shouldRunOnGPU.data), torch.cuda.FloatTensor)

if __name__ == '__main__':
    unittest.main()
