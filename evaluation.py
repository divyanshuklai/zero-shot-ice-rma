"""
evaluation.py – Evaluate a trained policy under (optionally) domain-randomized
conditions.  Structured like baseline.py: argparse, config dict, stats tracking
and saving, optional DR via TEST_CONFIG.  No reward scheduling.
"""
import os
import json
import numpy as np
from datetime import datetime

from environment import build_environment, reward_scheduler

# ── Go1 nominal PD gains ────────────────────────────────────────────────────
BASE_KP = 100
BASE_KD = 2

# ── Wider-than-training DR for stress testing ───────────────────────────────
TEST_CONFIG = {
    'resample_probability': 0.01,            # Resample every ~100 steps
    'friction': [0.04, 6.0],                 # Wider friction range
    'payload': [0.0, 7.0],                   # Heavier max payload
    'com': [-0.18, 0.18],                    # Further CoM shifts
    'motor_strength': [0.88, 1.22],          # More broken / overpowered motors
    'Kp': [BASE_KP * 0.8, BASE_KP * 1.2],   # [80, 120]
    'Kd': [BASE_KD * 0.5, BASE_KD * 1.5],   # [1.0, 3.0]
}

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    seed=42,
    num_episodes=100,
    flat=True,
    dr=False,
    friction=None,
    payload_scale=None,
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
    if args_dict.get("friction") is not None:
        tags.append(f"fric_{args_dict['friction']}")
    if args_dict.get("payload_scale") is not None:
        tags.append(f"pay_{args_dict['payload_scale']}")
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
        description="Evaluate a trained policy under configurable (optionally DR) conditions. "
                    "Generates per-episode stats, aggregate metrics, and an optional video rollout."
    )

    parser.add_argument("model_path", type=str,
                        help="Path to the saved model (.zip)")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                        help="Random seed for reproducibility")
    parser.add_argument("--num-episodes", type=int, default=DEFAULTS["num_episodes"],
                        help="Number of evaluation episodes")
    parser.add_argument("--flat", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["flat"],
                        help="Use flat terrain (--flat / --no-flat)")
    parser.add_argument("--dr", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["dr"],
                        help="Enable domain randomization with TEST_CONFIG (--dr / --no-dr)")
    parser.add_argument("--friction", type=float, default=DEFAULTS["friction"],
                        help="Manual floor friction override (ignores DR friction range)")
    parser.add_argument("--payload-scale", type=float, default=DEFAULTS["payload_scale"],
                        help="Manual payload scale applied to trunk mass (e.g. 1.1 = +10%%)")
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["record_video"],
                        help="Record a demo video of the final rollout")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["deterministic"],
                        help="Use deterministic actions during evaluation")
    parser.add_argument("--results-name", type=str, default=DEFAULTS["results_name"],
                        help="Results folder name (auto-generated if empty)")
    parser.add_argument("--log-dir", type=str, default=DEFAULTS["log_dir"],
                        help="Base directory for evals/ output")

    args = parser.parse_args()

    # Derive model_name from path
    args.model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    if not args.results_name:
        args_dict = {
            "seed": args.seed,
            "num_episodes": args.num_episodes,
            "flat": args.flat,
            "dr": args.dr,
            "friction": args.friction,
            "payload_scale": args.payload_scale,
            "record_video": args.record_video,
            "deterministic": args.deterministic,
            "results_name": args.results_name,
            "log_dir": args.log_dir,
        }
        args.results_name = get_default_results_name(args.model_name, args_dict)

    return args


# ── Environment builders ─────────────────────────────────────────────────────

def _no_schedule_reward():
    """Flat reward aggregation – just sum the components, no scheduling."""
    return reward_scheduler(
        scaling_function=lambda reward, *_args: np.sum(reward)
    )


def build_eval_env(args, seed_offset=0):
    """Build a single evaluation environment with optional DR / manual overrides."""
    env = build_environment(
        flat=args.flat,
        reward_scheduling=_no_schedule_reward(),
        randomize_domain=args.dr,
        randomization_params=TEST_CONFIG if args.dr else None,
        seed=args.seed + seed_offset,
    )

    # Manual friction override
    if args.friction is not None:
        env.unwrapped.model.geom_friction[0][0] = args.friction

    # Manual payload override
    if args.payload_scale is not None:
        env.unwrapped.model.body_mass[1] *= args.payload_scale

    return env


# ── Statistics helpers ────────────────────────────────────────────────────────

def compute_stats(episode_rewards, episode_lengths):
    """Compute aggregate statistics from evaluation episodes."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from gymnasium.wrappers import RecordVideo
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3 import PPO

    args = parse_args()
    set_random_seed(args.seed)

    evals_dir = os.path.join(args.log_dir, "evals")
    results_dir = os.path.join(evals_dir, args.results_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save eval config
    with open(os.path.join(results_dir, "eval_config.json"), "w") as f:
        cfg = vars(args).copy()
        cfg["test_config"] = TEST_CONFIG if args.dr else None
        json.dump(cfg, f, indent=2)

    # Load model
    env = build_eval_env(args)
    agent = PPO.load(args.model_path, env=env, device="cpu")
    vec_env = agent.get_env()

    # ── Evaluate episodes ─────────────────────────────────────────────────
    episodes = {}
    episode_rewards = []
    episode_lengths = []

    print(f"Evaluating {args.model_name} for {args.num_episodes} episodes "
          f"({'DR' if args.dr else 'no DR'}, "
          f"{'deterministic' if args.deterministic else 'stochastic'})…")

    for ep in range(args.num_episodes):
        obs = vec_env.reset()
        done = False
        step_count = 0
        ep_reward = 0.0

        actions = []
        rewards = []
        observations = [obs[0].tolist()]

        while not done:
            action, _state = agent.predict(obs, deterministic=args.deterministic)
            obs, rew, dones, info = vec_env.step(action)

            reward = rew[0]
            done = dones[0]
            step_count += 1
            ep_reward += reward

            actions.append(action[0].tolist())
            rewards.append(float(reward))
            observations.append(obs[0].tolist())

            print(f"\rEpisode {ep + 1}/{args.num_episodes}: "
                  f"{step_count} steps, reward = {ep_reward:.2f}", end="")

        print()

        episode_rewards.append(float(ep_reward))
        episode_lengths.append(step_count)

        episodes[f"episode_{ep}"] = {
            "actions": actions,
            "rewards": rewards,
            "observations": observations,
            "episode_reward": float(ep_reward),
            "episode_length": step_count,
        }

    # ── Aggregate statistics ──────────────────────────────────────────────
    stats = compute_stats(episode_rewards, episode_lengths)

    print(f"\n{'─' * 50}")
    print(f"Evaluation Summary  ({args.results_name})")
    print(f"{'─' * 50}")
    print(f"  Episodes        : {stats['num_episodes']}")
    print(f"  Mean reward     : {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Min / Max reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
    print(f"  Median reward   : {stats['median_reward']:.2f}")
    print(f"  Mean ep length  : {stats['mean_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
    print(f"  Success rate    : {stats['success_rate']:.1%}")
    print(f"  Survival rate   : {stats['survival_rate']:.1%}")
    print(f"{'─' * 50}")

    # Save results
    results_path = os.path.join(results_dir, f"{args.model_name}_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "test_config": TEST_CONFIG if args.dr else None,
            "stats": stats,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episodes": episodes,
        }, f, indent=2)
    print(f"Results saved to {results_path}")

    # ── Video recording ───────────────────────────────────────────────────
    if args.record_video:
        print("Recording demo video…")
        video_env = build_eval_env(args, seed_offset=1)
        video_env = RecordVideo(
            video_env,
            video_folder=results_dir,
            name_prefix=f"{args.model_name}_demo",
        )
        obs, _ = video_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _state = agent.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _info = video_env.step(action)
            total_reward += reward
            done = terminated or truncated
        video_env.close()
        print(f"Demo video reward: {total_reward:.2f}")

    vec_env.close()
    print("Done.")


if __name__ == "__main__":
    main()