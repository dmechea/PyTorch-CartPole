import random

def batchSample(sizeOfSample, currentGameMemory):
    return tuple(random.sample(currentGameMemory, sizeOfSample))

def addToGameMemory(MaxSize, sequenceToAdd, currentGameMemory):
    if currentGameMemory:
        return (*sequenceToAdd, *currentGameMemory)[0:MaxSize]
    else:
        return sequenceToAdd

def preProcessedMemory(state, prediction, actionTook, stepReward):
    return (state, prediction, actionTook, stepReward)

def addToSequence(currentMemoryStep, currentSequence=None):
    if currentSequence:
        return (currentMemoryStep, *currentSequence)
    else:
        return (currentMemoryStep, )

# turn the variable output to a tuple
def predictionToList(predictVariable):
    return list(predictVariable.data)

# Fix this
def modifyQValues(sequence, learningRate, discount, rePack):
    currentMem = None

    for index, memory in enumerate(sequence):
        state, Qvalues, action, reward = memory
        #most recent memory is terminal so nextQ = 0
        if index == 0:
            nextQ = 0

        Qvalues[action] = Qvalues[action] + learningRate*(
            reward + (discount*nextQ)
        )
        nextQ = Qvalues[action]

        currentMem = rePack(memory, currentMem)
    return currentMem
