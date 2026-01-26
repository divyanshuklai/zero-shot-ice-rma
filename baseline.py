"""
Method-0: baseline
RL algorithm used: PPO

This script trains a baseline policy on our training environment without any perturbations
to the observations and actions.

The baseline policy is then also tested on the testing environment.
"""
import gymnasium as gym
from evaluation import build_eval_env
from stable_baselines3.common.utils import set_random_seed

def train_env_builder(seed, rank=0):
    
    def _init():
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
            render_mode="rgb_array",
        )
        env.reset(seed=seed+rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    seed = 42
    n_envs = 6

    vec_env = SubprocVecEnv([train_env_builder(seed, env_n) for env_n in range(n_envs)])
    agent = PPO("MlpPolicy", vec_env, verbose=1)
    agent.learn(10_000_000, progress_bar=True)
    agent.save("models/PPO_1M_no_perturb.zip")
    vec_env.close()