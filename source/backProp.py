import torch.optim as optim
import torch.nn as nn

meanSquareError = nn.MSELoss()

def meansquareLoss(output, target):
    return meanSquareError(output, target)

def AdamOptimizer(neuralNetwork, learningRate):
    return optim.Adam(neuralNetwork.parameters(), lr=learningRate)

def SGD(neuralNetwork, learningRate):
    return optim.SGD(neuralNetwork.parameters(), lr=learningRate)

def backPropagate(
    Network,
    learningRate,
    optimizer,
    criteria,
    prediction,
    target,
    ):
    optimizer.zero_grad()
    loss = criteria(prediction, target)
    loss.backward()
    optimizer.step()
