import os
import torch
import gymnasium as gym
from environment import build_environment, reward_scheduler
from callbacks import ClampLogStdCallback

def train_env_builder(seed, rank=0):
    def _init():
        env = build_environment(
            flat=True,
            reward_scheduling=reward_scheduler(
                init_k=0.03,
                exponent=0.997
            )
        )
        env.reset(seed=seed+rank)
        return env
    return _init

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.utils import set_random_seed

    seed = 42
    set_random_seed(seed)

    n_iterations = 100
    batch_size_total = 80_000
    n_minibatches = 4
    n_epochs = 4
    n_envs = 40
    n_steps = batch_size_total // n_envs
    total_timesteps = n_iterations * batch_size_total
    minibatch_size = batch_size_total // n_minibatches

    model_name = "modal_run_100it_80kbs_rs03-997_run1"
    logging_dir = os.path.join("evals", model_name)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    vec_env = SubprocVecEnv([train_env_builder(seed, env_n) for env_n in range(n_envs)])

    policy_kwargs = dict(
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        log_std_init=-1.0,
    )

    agent = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=n_steps,
        batch_size=minibatch_size,
        n_epochs=n_epochs,
        clip_range=0.2,
        clip_range_vf=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        gae_lambda=0.95,
        gamma=0.998,
        learning_rate=5e-4,
        policy_kwargs=policy_kwargs,
        tensorboard_log=logging_dir,
        seed=seed,
    )

    agent.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        tb_log_name=model_name,
        callback=ClampLogStdCallback(min_std=0.2),
    )
    agent.save(f"models/{model_name}.zip")
    vec_env.close()