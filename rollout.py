import gym
import pybullet_envs
import numpy as np
from hebbian_neural_net import HebbianNet


def fitness(net: HebbianNet, env_name: str) -> float:
    env = gym.make(env_name)
    obs = env.reset()
    done = False
    r_tot = 0
    while not done:
        action = net.forward(obs)
        obs, r, done, _ = env.step(action)
        r_tot += r
    env.close()
    return r_tot
