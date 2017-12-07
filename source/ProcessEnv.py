import gym

env = gym.make('CartPole-v0')

env.reset()
goal_steps = 500

def some_random_games_first():
    for episode in range(100):
        env.reset()
        for t in range(goal_steps):
            action = env.action_space.sample()
    #        print (action)
            observation, reward, done, info = env.step(action)
            print (type(observation))
            if done:
                break

some_random_games_first()
