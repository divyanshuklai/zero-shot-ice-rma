"""
This script defines our common evaluation environment where we test our trained policies
"""
from gymnasium.wrappers import TransformReward
import gymnasium as gym
import numpy as np

def build_eval_env(friction, seed):
    from environment import build_environment

    env = build_environment(
        flat=True
    )

    #change conditions to icy (change sliding friction of floor(id:0))
    env.unwrapped.model.geom_friction[0][0] = friction
    #deterministically change weight of the agent +10%
    mass_var = 0.1
    env.unwrapped.model.body_mass[1] *= 1 + mass_var

    # print(f"world friction: {env.unwrapped.model.geom_friction}")
    # print(f"body mass : {env.unwrapped.model.body_mass}")
    return env

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Parse model to evaluate on OOD icy conditions." \
        "Generates video of a rollout, and statistics of evaluation." \
        "Returns mean reward and variance over 100 rollouts." \
        "Floor sliding friction is set to 0.1." \
        "Trunk body mass is randomized to +-10%."   
    )

    parser.add_argument("model_path", type=str, help="Path to the model, model_name is path base name.")
    parser.add_argument("-xs", "--exclude-stats", action="store_true", help="Only perform video rollouts.")
    parser.add_argument("-f", "--friction", type=float, default=0.1, help="Friction scaling of floor.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="default=42, Deterministic Inference. Randomization controlled thru seed.")
    parser.add_argument("-r", "--results-path", type=str, default=None, help="default=./evals/model_name")
    parser.add_argument("-e", "--num-episodes", type=int, default=100, help="number of episodes to evaluate.")
    
    from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3 import PPO
    import torch
    import json
    import os

    args = parser.parse_args()

    set_random_seed(args.seed)

    model_path = args.model_path
    model_name, _ = os.path.splitext(os.path.basename(model_path))
    
    results_path = os.path.join("./evals", model_name)
    if args.results_path:
        results_path = args.results_path
    
    os.makedirs(results_path, exist_ok=True)


    env = build_eval_env(args.friction, args.seed)
    env = RecordEpisodeStatistics(env, buffer_length=args.num_episodes)
    
    agent = PPO.load(model_path, env, device="cpu")
    vec_env = agent.get_env()

    episodes = {}

    if not args.exclude_stats:
        #eval
        for episode_num in range(args.num_episodes):
            obs = vec_env.reset()
            end = False
            step_count = 0
            episode_reward = 0

            actions = []
            rewards = []
            observations = [obs[0].tolist()]

            while not end:
                print("\r", end="")
                action, _state = agent.predict(obs, deterministic=True)
                obs, re, dones, info = vec_env.step(action)

                reward = re[0]
                end = dones[0]

                step_count += 1
                episode_reward += reward
                print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}", end="")

                actions.append(action[0].tolist())
                rewards.append(float(reward))
                observations.append(obs[0].tolist())
            print()
            
            episodes["episode_" + str(episode_num)] = {
                "actions":actions,
                "rewards":rewards,
                "observations":observations,
                "episode_reward":float(episode_reward)
            }

    # Final rollout with video recording
    print("recording video...")
    video_env = build_eval_env(args.friction, args.seed)
    video_env = RecordVideo(video_env, video_folder=results_path, name_prefix=str(model_name) + "_demo")
    obs, _ = video_env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _state = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = video_env.step(action)
        total_reward += reward
        done = terminated or truncated

    if not args.exclude_stats:
        print(f'\nEvaluation Summary:')
        print(f'Episode durations: {list(env.time_queue)}')
        print(f'Episode rewards: {list(env.return_queue)}')
        print(f'Episode lengths: {list(env.length_queue)}')

        #metrics
        avg_reward = float(np.mean(env.return_queue)) if len(env.return_queue) > 0 else 0.0
        avg_length = float(np.mean(env.length_queue)) if len(env.length_queue) > 0 else 0.0
        std_reward = float(np.std(env.return_queue)) if len(env.return_queue) > 0 else 0.0


        with open(os.path.join(results_path, model_name + "_results.json"), "w") as f:
            results = {
                "mean_reward":avg_reward,
                "std_reward":std_reward,
                "mean_episode_length":avg_length,
                "episodes":episodes
            }
            json.dump(results, f, indent=4)

        print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
        print(f'Average episode length: {avg_length:.1f} steps')
        # Fix: Handle empty case
        if len(env.return_queue) > 0:
            print(f'Success rate: {sum(1 for r in env.return_queue if r > 1000) / len(env.return_queue):.1%}')
        else:
            print('Success rate: N/A')

    video_env.close()
    vec_env.close()

if __name__ == "__main__":
    main()