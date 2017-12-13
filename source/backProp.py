import torch.optim as optim
import torch.nn as nn

def meanSquareError():
    return nn.MSELoss()

def AdamOptimizer(neuralNetwork, learningRate):
    return optim.Adam(neuralNetwork.parameters(), lr=learningRate)

def SGD(neuralNetwork, learningRate):
    return optim.SGD(neuralNetwork.parameters(), lr=learningRate)

def backPropagate(
    Network,
    optimizer,
    criteria,
    prediction,
    target,
    ):
    optimizer.zero_grad()
    loss = criteria(prediction, target)
    loss.backward()
    optimizer.step()
