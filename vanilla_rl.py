"""
Method-1: Vanilla RL
RL algorithm used: PPO 

We train a robot dog policy with domain randomization on:
    - friction: 
    - motor signal perturbations:
    - weight:

We show that vanilla training methods fail to work on transferrred environments (Sim2Sim)
thus set a baseline over which we will try to improve through teacher-student training.

We test our newly trained policy over 
"""
import gymnasium as gym
from stable_baselines3 import PPO

env = 
