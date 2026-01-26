"""
This script shows the environment and the robot we will be working with
"""
import gymnasium as gym
import numpy as np
from time import sleep

env = gym.make(
            "Ant-v5",
            xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
            forward_reward_weight=1,
            ctrl_cost_weight=0.05,
            contact_cost_weight=5e-4,
            healthy_reward=1,
            main_body=1,
            healthy_z_range=(0.195, 0.75),
            include_cfrc_ext_in_observation=True,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.1,
            frame_skip=25,
            max_episode_steps=1000,
            render_mode="human",
        )

obs, info = env.reset()
episode_over = False
step = 0
while not episode_over:
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    if step >= 1000 or truncated or terminated:
        episode_over = True
    env.render()
    sleep(1)
env.close()
