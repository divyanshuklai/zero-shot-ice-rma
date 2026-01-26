"""
This script defines our common evaluation environment where we test our trained policies
"""
import gymnasium as gym

def build_eval_env():
    #taken from https://gymnasium.farama.org/tutorials/gymnasium_basics/load_quadruped_model/
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
        render_mode="rgb_array",  # Change to "human" to visualize
        camera_name="tracking"
    )
    return env


if __name__ == "__main__":
    from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
    from stable_baselines3 import PPO
    from time import sleep
    import numpy as np
    import os

    model_path = "./models/PPO_1M_no_perturb"


    seed = 42
    env = build_eval_env()
    model_name, _ = os.path.splitext(os.path.basename(model_path))
    tgt_dir = os.path.join("./evals", model_name)
    if not os.path.exists(tgt_dir):
        os.mkdir(tgt_dir)
    env = RecordVideo(env, tgt_dir, name_prefix="eval", episode_trigger=lambda x: True)
    agent = PPO.load(model_path, env)
    vec_env = agent.get_env()
    obs = vec_env.reset()
    end = False
    num_steps = 0
    total_reward = 0
    while not end:
        action, _state = agent.predict(obs, deterministic=True)
        obs, reward, end, info = vec_env.step(action)
        total_reward += reward

    print(total_reward)
    env.close()