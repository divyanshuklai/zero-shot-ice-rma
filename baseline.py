import os
import json
import torch
from datetime import datetime
import gymnasium as gym
from environment import build_environment, reward_scheduler
from callbacks import ClampLogStdCallback, RewardScheduleCallback, AverageStepRewardCallback

# Go1 nominal PD gains
BASE_KP = 100
BASE_KD = 2

# RMA paper domain randomization config (scaled for Go1)
TRAIN_CONFIG = {
    'resample_probability': 0.004,  # Resample every ~250 steps
    'friction': [0.05, 4.5],        # Sliding friction coefficient
    'payload': [0.0, 6.0],          # Added mass in kg (Max ~50% body weight)
    'com': [-0.15, 0.15],           # Center of Mass displacement (meters)
    'motor_strength': [0.90, 1.10], # Torque multiplier (90% to 110%)

    # Scaled to Go1 Base Kp=100 (Paper used 55 +/- 5)
    'Kp': [BASE_KP * 0.9, BASE_KP * 1.1],  # [90, 110]

    # Scaled to Go1 Base Kd=2 (Paper used 0.6 +/- 0.2)
    'Kd': [BASE_KD * 0.67, BASE_KD * 1.33]  # [1.3, 2.7]
}

def get_default_model_name(args_dict, defaults_dict):
    """Generate model name as <#run>_DDMMYY_<params_changed>"""
    date_str = datetime.now().strftime("%d%m%y")
    
    changed_params = []
    skip_keys = {"model_name", "dr"}
    for key, default_val in defaults_dict.items():
        if key in skip_keys:
            continue
        if args_dict.get(key) != default_val:
            changed_params.append(f"{key}_{args_dict[key]}")
    
    dr_tag = "yes_dr" if args_dict.get("dr") else "no_dr"
    suffix = "_".join([dr_tag] + changed_params) if changed_params else dr_tag
    
    # Count existing runs today
    evals_dir = os.path.join(args_dict.get("log_dir", "."), "runs")
    os.makedirs(evals_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(evals_dir)
        if os.path.isdir(os.path.join(evals_dir, d)) and date_str in d
    ]
    run_number = len(existing) + 1
    
    return f"baseline_no_extrinsic_{run_number}_{date_str}_{suffix}"


DEFAULTS = dict(
    seed=42,
    n_iterations=100,
    batch_size_total=80_000,
    n_minibatches=4,
    n_epochs=4,
    n_envs=40,
    clip_range=0.2,
    clip_range_vf=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    gae_lambda=0.95,
    gamma=0.998,
    learning_rate=5e-4,
    log_std_init=-1.0,
    min_std=0.2,
    init_k=0.03,
    exponent=0.997,
    flat=True,
    dr=False,
    model_name="",
    log_dir=".",
)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run baseline training without privileged information. Configure hparams."
    )

    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Random seed")
    parser.add_argument("--n-iterations", type=int, default=DEFAULTS["n_iterations"], help="Number of training iterations")
    parser.add_argument("--batch-size-total", type=int, default=DEFAULTS["batch_size_total"], help="Total batch size per iteration")
    parser.add_argument("--n-minibatches", type=int, default=DEFAULTS["n_minibatches"], help="Number of minibatches")
    parser.add_argument("--n-epochs", type=int, default=DEFAULTS["n_epochs"], help="Number of PPO epochs per iteration")
    parser.add_argument("--n-envs", type=int, default=DEFAULTS["n_envs"], help="Number of parallel environments")
    parser.add_argument("--clip-range", type=float, default=DEFAULTS["clip_range"], help="PPO clip range")
    parser.add_argument("--clip-range-vf", type=float, default=DEFAULTS["clip_range_vf"], help="PPO value function clip range")
    parser.add_argument("--vf-coef", type=float, default=DEFAULTS["vf_coef"], help="Value function coefficient")
    parser.add_argument("--ent-coef", type=float, default=DEFAULTS["ent_coef"], help="Entropy coefficient")
    parser.add_argument("--gae-lambda", type=float, default=DEFAULTS["gae_lambda"], help="GAE lambda")
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma"], help="Discount factor")
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"], help="Learning rate")
    parser.add_argument("--log-std-init", type=float, default=DEFAULTS["log_std_init"], help="Initial log std for policy")
    parser.add_argument("--min-std", type=float, default=DEFAULTS["min_std"], help="Minimum std for ClampLogStdCallback")
    parser.add_argument("--init-k", type=float, default=DEFAULTS["init_k"], help="Initial k for reward scheduling")
    parser.add_argument("--exponent", type=float, default=DEFAULTS["exponent"], help="Exponent for reward scheduling")
    parser.add_argument("--flat", action=argparse.BooleanOptionalAction, default=DEFAULTS["flat"], help="Use flat terrain (--flat / --no-flat)")
    parser.add_argument("--dr", action=argparse.BooleanOptionalAction, default=DEFAULTS["dr"], help="Enable RMA domain randomization (--dr / --no-dr)")
    parser.add_argument("--model-name", type=str, default=DEFAULTS["model_name"], help="Model name (auto-generated if empty)")
    parser.add_argument("--log-dir", type=str, default=DEFAULTS["log_dir"], help="Base directory for runs/ and models/ output")

    args = parser.parse_args()

    # Build args dict with matching keys to DEFAULTS
    args_dict = {
        "seed": args.seed,
        "n_iterations": args.n_iterations,
        "batch_size_total": args.batch_size_total,
        "n_minibatches": args.n_minibatches,
        "n_epochs": args.n_epochs,
        "n_envs": args.n_envs,
        "clip_range": args.clip_range,
        "clip_range_vf": args.clip_range_vf,
        "vf_coef": args.vf_coef,
        "ent_coef": args.ent_coef,
        "gae_lambda": args.gae_lambda,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "log_std_init": args.log_std_init,
        "min_std": args.min_std,
        "init_k": args.init_k,
        "exponent": args.exponent,
        "flat": args.flat,
        "dr": args.dr,
        "model_name": args.model_name,
        "log_dir": args.log_dir,
    }

    if not args.model_name:
        args.model_name = get_default_model_name(args_dict, DEFAULTS)

    return args


def train_env_builder(args, rank=0):
    def _init():
        env = build_environment(
            flat=args.flat,
            reward_scheduling=reward_scheduler(
                init_k=args.init_k,
                exponent=args.exponent
            ),
            randomize_domain=args.dr,
            randomization_params=TRAIN_CONFIG if args.dr else None,
        )
        env.reset(seed=args.seed+rank)
        return env
    return _init

def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.utils import set_random_seed

    args = parse_args()
    set_random_seed(args.seed)

    n_steps = args.batch_size_total // args.n_envs
    total_timesteps = args.n_iterations * args.batch_size_total
    minibatch_size = args.batch_size_total // args.n_minibatches

    runs_dir = os.path.join(args.log_dir, "runs")
    models_dir = os.path.join(args.log_dir, "models")
    logging_dir = os.path.join(runs_dir, args.model_name)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(logging_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    vec_env = SubprocVecEnv([train_env_builder(args, env_n) for env_n in range(args.n_envs)])

    policy_kwargs = dict(
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        log_std_init=args.log_std_init,
    )

    agent = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=n_steps,
        batch_size=minibatch_size,
        n_epochs=args.n_epochs,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
        tensorboard_log=logging_dir,
        seed=args.seed,
    )

    agent.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        tb_log_name="PPO",
        callback=[
            ClampLogStdCallback(min_std=args.min_std),
            RewardScheduleCallback(init_k=args.init_k, exponent=args.exponent),
            AverageStepRewardCallback(),
        ],
    )
    agent.save(os.path.join(models_dir, f"{args.model_name}.zip"))
    vec_env.close()

if __name__ == "__main__":
    main()