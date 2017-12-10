import random

def batchSample(sizeOfSample, currentGameMemory):
    return tuple(random.sample(currentGameMemory, sizeOfSample))

def isFull(maxSizeAllowed, currentGameMemory):
    return len(currentGameMemory) >= maxSizeAllowed

def addToGameMemory(isMaxSize, memoryToAdd, currentGameMemory):
    if isMaxSize:
        return (memoryToAdd, *currentGameMemory)[0:-1]
    else:
        return (memoryToAdd, *currentGameMemory)
