import torch
from torch.autograd import Variable
from Networks import LinearNet
import numpy as np
import random

#takes a numpy array and returns a torch Variable
def convertToVariable(state):
    return Variable(torch.from_numpy(state).type(FloatTensor))

#takes a network and an input environment / state and returns a prediction
def makePrediction(network, state):
    return network(state)

#Takes a FloatTensor with output predictions and returns action integer
def chooseMaxQVal(prediction):
    predictionValue, index = prediction.data.max(0)
    return list(index)[0]

#Takes the initial and final sampling ratio and the number of steps to decay
#Returns the step size
def explorationDecay(initialExploration, finalExploration, episodeNumbers):
    return (initialExploration - finalExploration) / episodeNumbers

def updatedExploration(currentExploration, stepReduction):
    return currentExploration - stepReduction

def randomAction(actionsAvailable):
    return LongTensor([[random.randrange(actionsAvailable)]])

def selectAction(network, state, exploration):
    sample = random.random()
    if sample > exploration:
        predict = makePrediction(network, state)
        chosenAction = chooseMaxQVal(predict)
    else:
        randomAction(2)
