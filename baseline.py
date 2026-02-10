"""
Method-0: baseline
RL algorithm used: PPO

This script trains a baseline policy on our training environment without any domain radomization

The baseline policy is then also tested on the evaluation environment.
"""
import gymnasium as gym
from environment import build_environment, reward_scheduler

def train_env_builder(seed, rank=0):
    def _init():
        env = build_environment(
            reward_scheduling=reward_scheduler()
        )
        env.reset(seed=seed+rank)
        return env
    return _init

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    seed = 42
    n_envs = 6
    set_random_seed(seed)

    vec_env = SubprocVecEnv([train_env_builder(seed, env_n) for env_n in range(n_envs)])
    agent = PPO("MlpPolicy", vec_env, verbose=1)
    agent.learn(3_000_000, progress_bar=True)
    agent.save("models/PPO_3M_STABLE_GAIT_REWARDS_10FEB_RMA_SCHEDULE.zip")
    vec_env.close()