import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import ActionSpace as act
import backProp as bp
import gameMemory as mem
import Networks as NN
import FeaturesLabels as FL
import Cuda as GPU

env = gym.make('CartPole-v0')

env.reset()
framesPerEpisode = 200
episodes = 2000

numActions = 2
Net = NN.LinearThreeDeep(4, 20, 30, 20, numActions)
Net = GPU.assignToGPU(Net)
OptimizerLearningRate = 0.003
####################
# Loss used
criterion = nn.MSELoss()
####################
# Optimizer Used
optimizer = bp.AdamOptimizer(Net, OptimizerLearningRate)

def playTheGame(isGPU = False):
    FloatTensor = torch.cuda.FloatTensor if isGPU else torch.FloatTensor
    memorySize = 2000
    memory = None
    learningBatch = []

    QlearningRate = 1
    discountFactor = 0.99

    begunTraining = False

    currentExploration = 0.9
    finalExplore = 0.05
    expEpisodes = 1500
    expDecay = act.explorationDecay(
        currentExploration,
        finalExplore,
        expEpisodes,
    )

    topSessions = []
    gameScoreBenchmark = 0

    last100Games = []
    last100Averages = []
    countNegativeDiffs = []


    for episode in range(episodes):
        env.reset()
        #zero the state
        prev_state = np.array([0,0,0,0])

        #reset the sequence
        currentSequence = None

        EpisodeScore = 0
        actionAllocation = [0,0]

        for frame in range(framesPerEpisode):
            ##########################################
            # BEGIN EPISODE
            if episode > 1990:
                env.render()
            ##########################################
            # Make a Prediction
            prediction = act.makePrediction(Net, prev_state, isGPU)

            ##########################################
            # Perform an action
            action = act.selectAction(
                numActions,
                prediction,
                currentExploration,
                isGPU,
            )
            actionAllocation[action] += 1

            ###########################################
            # Get Results of Action Taken
            state, reward, done, info = env.step(action)

            EpisodeScore += reward
            ############################################################
            # Store individual State Memory and add it to the sequence
            formattedPrediction = mem.predictionToList(prediction)
            state_memory = mem.preProcessedMemory(
                prev_state,
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
        # Update Exploitation vs Exploration Ratio
        if begunTraining:
            currentExploration = act.updatedExploration(
                currentExploration,
                finalExplore,
                expDecay,
            )


        BenchmarkSize = 100
        gameScoreBenchmark = mem.getMean(topSessions)
        if EpisodeScore >= gameScoreBenchmark:
            topSessions = mem.addToTopSessions(BenchmarkSize, EpisodeScore, topSessions)

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

        last100Games = mem.addToLast100(EpisodeScore, last100Games)
        rolling100Average = mem.getMean(last100Games)
        last100Averages = mem.addToLast100(rolling100Average, last100Averages)


        if len(last100Averages) > 2:
            averageDiff = (last100Averages[0] - last100Averages[-1])
        else:
            averageDiff = 0

        if episode % 100 == 0:
            print ('Episode Number:', episode,
                'Game Score:', EpisodeScore,
                'BenchMark:', int(gameScoreBenchmark),
                'AV:', int(rolling100Average),
                'Diff:', int(averageDiff),
                'EXP:', currentExploration)

        #####################################################
        # Learning Zone
        batchSize = 10
        #Criteria to do a learningLoop
        epochs = 5
#        print (episode % 1000, episode)
        if episode % 250 == 0 and episode > 400:
            begunTraining = True
            print ('Begin Training')
            print ('Memory Size: ', len(memory))
            print ('Batch Size: ', batchSize)
            print ('Sample Rounds: ', int(len(memory)/batchSize))
            print ('Epochs: ', epochs)
            sampleRounds = int(len(memory)/batchSize)
            for epoch in range(epochs):
                for batch in range(sampleRounds):

                    learningBatch = mem.batchSample(batchSize, memory)
                    # Make the features and labels
                    Features, Labels = FL.memoryToFeatureLabel(memory)
                    netTrainingPredict = act.makePrediction(Net, Features, isGPU)
                    VLabels = act.convertToVariable(Labels, FloatTensor)
                    optimizer.zero_grad()
                    loss = criterion(netTrainingPredict, VLabels)
                    loss.backward()
                    optimizer.step()

                    if batch % 100 == 0:

                        print ('Epoch: ', epoch,
                                'Batch: ', batch, '/', sampleRounds,
                                'Loss: ', loss.data[0])
            memory = None

playTheGame(isGPU=True)
