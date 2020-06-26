import gym
import numpy as np
import random
from collections import defaultdict

#generate episodes
#select actions with erp
#store all those episodes

#algo
#FIMCA

env= gym.make('FrozenLake-v0')
action_space = env.action_space.n
obs_space = env.observation_space.n

# print(action_space, obs_space)

def generate_episode(env):
    episode = []
    state = env.reset()
    while True:
        #ERP
        probs  = [0.25,0.25,0.25,0.25]
        action = np.random.choice(np.arange(4), p= probs)
        next_state, reward, done, info = env.step(action)
        episode.append((state,action,reward))
        state = next_state
        if done:
            break
    return episode

for i in range(100):
    print("{} \n {}episode".format(generate_episode(env), i))

def first_visit_MCM(env, number_episodes, generate_episode, gamma=0.9):
    all_return_sum = defaultdict(lambda: np.zeros(action_space))
    N = defaultdict(lambda: np.zeros(action_space))
    Q = defaultdict(lambda: np.zeros(action_space))
    for epi in range(number_episodes):
        episode = generate_episode(env)
        states, actions, rewards = zip(*episode)
        discounted_sum = np.array([gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            all_return_sum[state][actions[i]] += sum(rewards[i:]*discounted_sum[:-(1+i)])
            N[state][actions[i]] += 1
            Q[state][actions[i]] = all_return_sum[state][actions[i]] / N[state][actions[i]]
    return Q
