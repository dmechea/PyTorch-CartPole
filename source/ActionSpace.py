import torch
from torch.autograd import Variable
import numpy as np
import random

#takes a numpy array and returns a torch Variable
def convertToVariable(state, tensorType):
    return Variable(torch.from_numpy(state).type(tensorType))

#takes a network and an input environment / state and returns a prediction
def makePrediction(network, state):
    return network(state)

#Takes a FloatTensor with output predictions and returns action integer
def chooseMaxQVal(prediction):
    predictionValue, index = prediction.data.max(0)
    return list(index)[0]

#Takes the initial and final exploration ratio and the number of steps to decay
#eg 0.9 = 90% random actions, 10% prediction actions
#Returns the step size
def explorationDecay(initialExploration, finalExploration, episodeNumbers):
    return (initialExploration - finalExploration) / episodeNumbers

def updatedExploration(currentExp, finalExp, step):
    return finalExp if finalExp >= currentExp else currentExp - step

def randomAction(actionsAvailable, tensorType):
    guessTensor = tensorType([[random.randrange(actionsAvailable)]])
    return int(guessTensor[0][0])

# givens a random sample between 0 & 1
def getRandomSample():
    return random.random()

#where guess = randomAction, sample is a getRandomSample()
def selectAction(sample, exploration, predictChoice, guess):
    return predictChoice if sample > exploration else guess
