"""
evaluation.py - evaluate policy on slippery test world.
"""

import os
import json
import jax
import jax.numpy as jp
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from environment import build_environment
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.io import model as brax_model

DEFAULTS = dict(
    seed=42,
    num_episodes=100,
    flat=True,
    dr=False,
    slippery=False,
    record_video=True,
    deterministic=True,
    results_name="",
    log_dir=".",
)

def get_default_results_name(model_name: str, args_dict: dict) -> str:
    """Generate results folder name: eval_<model>_<#run>_DDMMYY_<params_changed>"""
    date_str = datetime.now().strftime("%d%m%y")

    tags = []
    tags.append("dr" if args_dict.get("dr") else "no_dr")
    if args_dict.get("slippery"):
        tags.append("slippery")
    if not args_dict.get("flat"):
        tags.append("terrain")

    suffix = "_".join(tags)

    evals_dir = os.path.join(args_dict.get("log_dir", "."), "evals")
    os.makedirs(evals_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(evals_dir)
        if os.path.isdir(os.path.join(evals_dir, d)) and date_str in d
    ]
    run_number = len(existing) + 1

    return f"eval_{model_name}_{run_number}_{date_str}_{suffix}"


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MJX policy."
    )

    parser.add_argument("model_path", type=str,
                        help="Path to the saved model parameters")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                        help="Random seed")
    parser.add_argument("--num-episodes", type=int, default=DEFAULTS["num_episodes"],
                        help="Number of evaluation episodes")
    parser.add_argument("--flat", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["flat"],
                        help="Use flat terrain")
    parser.add_argument("--dr", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["dr"],
                        help="Enable domain randomization")
    parser.add_argument("--slippery", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["slippery"],
                        help="Use slippery terrain (requires DR to be enabled in env config)")
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["record_video"],
                        help="Record a demo video (requires mujoco rendering)")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["deterministic"],
                        help="Use deterministic actions")
    parser.add_argument("--results-name", type=str, default=DEFAULTS["results_name"],
                        help="Results folder name")
    parser.add_argument("--log-dir", type=str, default=DEFAULTS["log_dir"],
                        help="Base directory for output")

    args = parser.parse_args()

    args.model_name = os.path.basename(args.model_path)

    if not args.results_name:
        args_dict = vars(args)
        args.results_name = get_default_results_name(args.model_name, args_dict)

    return args


def compute_stats(episode_rewards, episode_lengths):
    """Compute aggregate statistics."""
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)

    return {
        "num_episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "median_reward": float(np.median(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "std_episode_length": float(np.std(lengths)),
        "min_episode_length": int(np.min(lengths)),
        "max_episode_length": int(np.max(lengths)),
        "success_rate": float(np.mean(rewards > 1000)),
        "survival_rate": float(np.mean(lengths >= 1000)),
    }


def main():
    args = parse_args()
    
    evals_dir = os.path.join(args.log_dir, "evals")
    results_dir = os.path.join(evals_dir, args.results_name)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "eval_config.json"), "w") as f:
        cfg = vars(args).copy()
        json.dump(cfg, f, indent=2)

    env = build_environment(
        flat=args.flat,
        randomize_domain=args.dr,
        slippery_eval=args.slippery,
        init_k=0.03,  # During eval, maybe use a fixed k or 0.0?
        exponent=1.0,  # Keep k constant during eval
        seed=args.seed,
        num_envs=1,
        reward_schedule_steps_per_iteration=None,
    )

    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size
    )
    params = brax_model.load_params(args.model_path)
    
    inference_fn = ppo_networks.make_inference_fn(ppo_network)(
        params, deterministic=args.deterministic
    )
    jit_inference_fn = jax.jit(inference_fn)
    jit_step_fn = jax.jit(env.step)
    
    @jax.jit
    def batched_reset(rng):
        rng = jax.random.split(rng, 1)
        return env.reset(rng)

    print(f"Evaluating {args.model_name} for {args.num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    rng = jax.random.PRNGKey(args.seed)
    
    for ep in range(args.num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state = batched_reset(reset_rng)
        
        ep_reward = 0.0
        step_count = 0
        
        while not np.all(state.done):
            rng, action_rng = jax.random.split(rng)
            action_rng = jax.random.split(action_rng, 1) # batched rng
            
            action, _ = jit_inference_fn(state.obs, action_rng)
            state = jit_step_fn(state, action)
            
            ep_reward += float(state.reward[0])
            step_count += 1
            
            if step_count >= 1000:
                break
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(step_count)
        print(f"Episode {ep+1}/{args.num_episodes}: Reward={ep_reward:.2f}, Length={step_count}")

    stats = compute_stats(episode_rewards, episode_lengths)
    
    print(f"\nEvaluation Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "stats": stats,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }, f, indent=2)

if __name__ == "__main__":
    main()
