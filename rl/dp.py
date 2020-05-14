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
plot_values(V)



