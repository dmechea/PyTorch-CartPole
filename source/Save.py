import torch
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def saveParameters(network, fileName):
    return torch.save(
        network.state_dict(),
        os.path.join(dir_path, 'Saves', fileName))

def loadParameters(network, fileName):
    return network.load_state_dict(
        torch.load(os.path.join(dir_path, 'Saves', fileName)))
