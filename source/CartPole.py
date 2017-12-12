import gym
import numpy as np
import torch

import ActionSpace as act
import backProp as bp
import gameMemory as mem
import Networks as NN
import FeaturesLabels as FL

env = gym.make('CartPole-v0')

env.reset()
framesPerEpisode = 500
episodes = 1000

numActions = 2
Net = NN.LinearTwoDeep(4, 20, 20, numActions)

batchSize = 200



def playTheGame():

    memorySize = 100000
    memory = None
    learningBatch = []

    QlearningRate = 0.1
    discountFactor = 0.99

    currentExploration = 0.9
    finalExplore = 0.05
    expEpisodes = 500
    expDecay = act.explorationDecay(
        currentExploration,
        finalExplore,
        expEpisodes,
    )

    for episode in range(episodes):
        env.reset()
        #zero the state
        prev_state = np.array([0,0,0,0])

        #reset the sequence
        currentSequence = None

        for frame in range(framesPerEpisode):
            ##########################################
            # BEGIN EPISODE

            ##########################################
            # Make a Prediction
            prediction = act.makePrediction(Net, prev_state)

            ##########################################
            # Perform an action
            action = act.selectAction(numActions, prediction, currentExploration)

            ###########################################
            # Get Results of Action Taken
            state, reward, done, info = env.step(action)

            ############################################################
            # Store individual State Memory and add it to the sequence
            formattedPrediction = mem.predictionToList(prediction)
            state_memory = mem.preProcessedMemory(
                state,
                formattedPrediction,
                action,
                reward,
            )
            currentSequence = mem.addToSequence(state_memory, currentSequence)

            prev_state = state

            if done:
                env.reset()
                break

        ##################################
        # POST EPISODE

        #####################################################
        # Calculated and update Q Values due to result experience
        QValuedSequenceMemory = mem.modifyQValues(
            currentSequence,
            QlearningRate,
            discountFactor,
            mem.addToSequence,
        )

        #####################################################
        # Add Sequence to Game Memory
        memory = mem.addToGameMemory(
            memorySize,
            QValuedSequenceMemory,
            memory
        )

        #####################################################
        # Update Exploitation vs Exploration Ratio
        currentExploration = act.updatedExploration(
            currentExploration,
            finalExplore,
            expDecay,
        )

        if len(memory) > batchSize:
            learningBatch = mem.batchSample(batchSize, memory)


        # Make the features and labels
        Features, Labels = FL.memoryToFeatureLabel(learningBatch)

        print (Features, Labels)

    print (len(learningBatch))




playTheGame()
