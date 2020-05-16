import numpy as np
import check_test
from frozenlake import FrozenLakeEnv
from plot_utils import plot_values

env = FrozenLakeEnv()

#LEFT = 0
#RIGHT = 2
#UP = 3
#DOWN = 1

#[[0 1 2 3]
# [4 5 6 7]
# [8 9 10 11]
# [12 13 14 15]]]

print(env.observation_space)
print(env.action_space)

sp = env.nS
a_s = env.nA

def policy_eval(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V

random_policy = np.ones([sp,a_s])/a_s
V = policy_eval(env, random_policy)

#Action value function
def action_value_function(env, V, s, gamma=0.9):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

Q = np.zeros([sp, a_s])
for s in range(sp):
    Q[s] = action_value_function(env, V, s)
print(Q)

def new_policy(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q = action_value_function(env, V, s, gamma)
        best_a = np.argwhere(q==np.max(q)).flatten()
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)
    return policy
policy = new_policy(env, V) 
print('/n new policy is:', policy, 'L=0, R = 2, U= 3, D= 1')





