import numpy as np

def memoryToFeatureLabel(memoryBatch):
    featureSet = []
    labelSet = []

    for memory in memoryBatch:
        feature = memory[0]
        label = memory [1]

        featureSet.append(feature)
        labelSet.append(label)

    return (np.array(featureSet), np.array(labelSet))
