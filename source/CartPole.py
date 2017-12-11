import gym
import numpy as np
import torch

import ActionSpace as act
import backProp as bp
import gameMemory as mem
import Networks as NN

env = gym.make('CartPole-v0')

env.reset()
goal_steps = 500
episodes = 1000


Net = NN.LinearTwoDeep(4, 20, 20, 2)

def playTheGame():

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

        for frame in range(goal_steps):
            env.render()
            sample = act.getRandomSample()
            guess = act.randomAction(2, torch.LongTensor)
            inputFeed = act.convertToVariable(
                prev_state,
                torch.FloatTensor,
            )
            prediction = act.makePrediction(Net, inputFeed)
            maxQAction = act.chooseMaxQVal(prediction)

            action = act.selectAction(
                sample,
                currentExploration,
                maxQAction,
                guess,
            )
            state, reward, done, info = env.step(action)
            prev_state = state

            

            if done:
                break

        currentExploration = act.updatedExploration(
            currentExploration,
            finalExplore,
            expDecay,
        )

playTheGame()
