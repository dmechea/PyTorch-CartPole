import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputLinear = nn.Linear(inputSize, hiddenSize)
        self.outputLinear = nn.Linear(hiddenSize, outputSize)

    def forward(self, inputNodes):
        hiddenNodes = F.relu(self.inputLinear(inputNodes))
        outputNodes = F.softmax(self.outputLinear(hiddenNodes))
        return outputNodes

    def getParameters(self):
        return tuple(self.parameters())

#a = np.array([100,90,78,0.3])
#b = np.array([-0.5,0.54,-0.98,0.23])
#c = np.array([-5000,-43241,-944,-23])
#d = np.array([a,b,c])
#x = Variable(torch.from_numpy(d).type(torch.FloatTensor))
#inputsss = Variable(torch.randn(4))
