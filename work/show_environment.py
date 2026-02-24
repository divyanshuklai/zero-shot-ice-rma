"""
This script shows the environment and the robot we will be working with
"""
import gymnasium as gym
import numpy as np
from time import sleep

from environment import build_environment

env = build_environment()
obs, info = env.reset()
episode_over = False
step = 0

print(info, "\n",obs)

while not episode_over:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    if step >= 1000 or truncated or terminated:
        episode_over = True
    env.render()
    sleep(0.01)
env.close()
