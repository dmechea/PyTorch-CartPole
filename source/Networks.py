import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

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

class LinearTwoDeep(nn.Module):
    def __init__(self,
            inputSize,
            hiddenSize1,
            hiddenSize2,
            outputSize):
        super().__init__()
        self.inputLinear = nn.Linear(inputSize, hiddenSize1)
        self.hiddenLinear = nn.Linear(hiddenSize1, hiddenSize2)
        self.outputLinear = nn.Linear(hiddenSize2, outputSize)

    def forward(self, inputNodes):
        hiddenNodesL1 = F.relu(self.inputLinear(inputNodes))
        hiddenNodesL2 = F.relu(self.hiddenLinear(hiddenNodesL1))
        outputNodes = F.softmax(self.outputLinear(hiddenNodesL2))
        return outputNodes

    def getParameters(self):
        return tuple(self.parameters())

class LinearThreeDeep(nn.Module):
    def __init__(self,
            inputSize,
            hiddenSize1,
            hiddenSize2,
            hiddenSize3,
            outputSize,
        ):
        super().__init__()
        self.inputLinear = nn.Linear(inputSize, hiddenSize1)
        self.hiddenLinear1 = nn.Linear(hiddenSize1, hiddenSize2)
        self.hiddenLinear2 = nn.Linear(hiddenSize2, hiddenSize3)
        self.outputLinear = nn.Linear(hiddenSize3, outputSize)

    def forward(self, inputNodes):
        hiddenNodesL1 = F.relu(self.inputLinear(inputNodes))
        hiddenNodesL2 = F.relu(self.hiddenLinear1(hiddenNodesL1))
        hiddenNodesL3 = F.relu(self.hiddenLinear2(hiddenNodesL2))
        outputNodes = F.softmax(self.outputLinear(hiddenNodesL3))
        return outputNodes

    def getParameters(self):
        return tuple(self.parameters())
