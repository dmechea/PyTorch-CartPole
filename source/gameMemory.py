import random

def batchSample(sizeOfSample, currentGameMemory):
    return tuple(random.sample(currentGameMemory, sizeOfSample))

def maxMemorySize(maxSizeAllowed, currentGameMemory):
    return len(currentGameMemory) >= maxSizeAllowed

def addToGameMemory(maxSize, memoryToAdd, currentGameMemory):
    if maxMemorySize(maxSize, currentGameMemory):
        return (memoryToAdd, *currentGameMemory)[0:-1]
    else:
        return (memoryToAdd, *currentGameMemory)
