"""
rma_phase1 --> training teacher policy and priviledged extrinsics encoder.
"""

import os
import json
import jax
from datetime import datetime
from typing import Dict, Any, Optional

from environment import build_environment
from brax.envs import base as brax_env
import jax.numpy as jnp
from brax.io import model as brax_model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

class TeacherObservationWrapper(brax_env.Wrapper):
    """
    Concatenates proprioceptive and privileged observations for PPO.
    The Teacher policy in the network will split them back.
    """
    def __init__(self, env):
        super().__init__(env)
        # observation_size might be a tuple or dict depending on wrapping
        self.proprio_size = 48
        self.privileged_size = 123

    @property
    def observation_size(self):
        return self.proprio_size + self.privileged_size

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        # state.obs is already a dict from RMAWrapper
        obs = jnp.concatenate([state.obs, state.info["privileged_state"]], axis=-1)
        return state.replace(obs=obs)

    def step(self, state, action):
        # Inner env might use scan (e.g. ActionRepeatWrapper).
        # We must restore the obs shape to what inner env expects to avoid scan carry shape mismatch.
        state = state.replace(obs=state.obs[..., :self.proprio_size])
        state = self.env.step(state, action)
        obs = jnp.concatenate([state.obs, state.info["privileged_state"]], axis=-1)
        return state.replace(obs=obs)

def make_teacher_ppo_networks(
    observation_size: int,
    action_size: int,
    
    proprio_size: int = 48,
    latent_size: int = 8,
):
    from rma import TeacherEncoder, TeacherNetwork
    from brax.training import types, distribution
    from brax.training.networks import FeedForwardNetwork
    import flax.linen as nn
    import jax.numpy as jnp

    parametric_action_distribution = distribution.NormalDistribution(event_size=action_size)

    class TeacherPolicy(nn.Module):
        @nn.compact
        def __call__(self, x):
            proprio = x[..., :proprio_size]
            privileged = x[..., proprio_size:]
            
            z = TeacherEncoder(latent_size=latent_size)(privileged)
            logits, _ = TeacherNetwork(action_size=action_size)(proprio, z)
            
            log_std = self.param('log_std', nn.initializers.constant(-1.0), (action_size,))
            std = jnp.exp(log_std)
            std = jnp.broadcast_to(std, logits.shape)
            
            return logits, std

    class TeacherValue(nn.Module):
        @nn.compact
        def __call__(self, x):
            proprio = x[..., :proprio_size]
            privileged = x[..., proprio_size:]
            
            z = TeacherEncoder(latent_size=latent_size)(privileged)
            _, value = TeacherNetwork(action_size=action_size)(proprio, z)
            
            return value

    policy_module = TeacherPolicy()
    value_module = TeacherValue()
    
    if isinstance(observation_size, int):
        obs_shape = (observation_size,)
    else:
        obs_shape = tuple(observation_size)
    dummy_obs = jnp.zeros((1,) + obs_shape)

    def policy_init(key):
        return policy_module.init(key, dummy_obs)

    def policy_apply(processor_params, policy_params, obs):
        return policy_module.apply(policy_params, obs)

    def value_init(key):
        return value_module.init(key, dummy_obs)

    def value_apply(processor_params, value_params, obs):
        return value_module.apply(value_params, obs)

    policy_network = FeedForwardNetwork(init=policy_init, apply=policy_apply)
    value_network = FeedForwardNetwork(init=value_init, apply=value_apply)

    return ppo_networks.PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

DEFAULTS = dict(
    seed=42,
    n_iterations=100,
    batch_size_total=80_000,
    n_minibatches=4,
    n_epochs=4,
    n_envs=4096, 
    clip_range=0.2,
    clip_range_vf=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    gae_lambda=0.95,
    gamma=0.998,
    learning_rate=5e-4,
    log_std_init=-1.0,
    init_k=0.03,
    exponent=0.997,
    flat=False,
    dr=False,
    model_name="",
    log_dir=".",
)

def get_default_model_name(args_dict, defaults_dict):
    """Generate model name as <#run>_DDMMYY_<params_changed>"""
    date_str = datetime.now().strftime("%d%m%y")

    def _safe(v):
        return (
            str(v)
            .replace("/", "-")
            .replace("\\", "-")
            .replace(" ", "")
            .replace(":", "-")
        )
    
    changed_params = []
    skip_keys = {"model_name", "dr", "log_dir"}
    for key, default_val in defaults_dict.items():
        if key in skip_keys:
            continue
        if args_dict.get(key) != default_val:
            changed_params.append(f"{key}_{_safe(args_dict[key])}")
    
    dr_tag = "yes_dr" if args_dict.get("dr") else "no_dr"
    suffix = "_".join([dr_tag] + changed_params) if changed_params else dr_tag
    
    evals_dir = os.path.join(args_dict.get("log_dir", "."), "runs")
    os.makedirs(evals_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(evals_dir)
        if os.path.isdir(os.path.join(evals_dir, d)) and date_str in d
    ]
    run_number = len(existing) + 1
    
    return f"rma_phase1_mjx_{run_number}_{date_str}_{suffix}"


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run baseline training with MJX. Configure hparams."
    )

    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--n-iterations", type=int, default=DEFAULTS["n_iterations"])
    parser.add_argument("--batch-size-total", type=int, default=DEFAULTS["batch_size_total"])
    parser.add_argument("--n-minibatches", type=int, default=DEFAULTS["n_minibatches"])
    parser.add_argument("--n-epochs", type=int, default=DEFAULTS["n_epochs"])
    parser.add_argument("--n-envs", type=int, default=DEFAULTS["n_envs"])
    parser.add_argument("--clip-range", type=float, default=DEFAULTS["clip_range"])
    parser.add_argument("--clip-range-vf", type=float, default=DEFAULTS["clip_range_vf"])
    parser.add_argument("--vf-coef", type=float, default=DEFAULTS["vf_coef"])
    parser.add_argument("--ent-coef", type=float, default=DEFAULTS["ent_coef"])
    parser.add_argument("--gae-lambda", type=float, default=DEFAULTS["gae_lambda"])
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--log-std-init", type=float, default=DEFAULTS["log_std_init"])
    parser.add_argument("--init-k", type=float, default=DEFAULTS["init_k"])
    parser.add_argument("--exponent", type=float, default=DEFAULTS["exponent"])
    parser.add_argument("--flat", action=argparse.BooleanOptionalAction, default=DEFAULTS["flat"])
    parser.add_argument("--dr", action=argparse.BooleanOptionalAction, default=DEFAULTS["dr"])
    parser.add_argument("--model-name", type=str, default=DEFAULTS["model_name"])
    parser.add_argument("--log-dir", type=str, default=DEFAULTS["log_dir"])

    args = parser.parse_args()
    args_dict = vars(args)
    if not args.model_name:
        args.model_name = get_default_model_name(args_dict, DEFAULTS)
    return args


def main():
    args = parse_args()
    
    runs_dir = os.path.join(args.log_dir, "runs")
    models_dir = os.path.join(args.log_dir, "models")
    logging_dir = os.path.join(runs_dir, args.model_name)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(logging_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    unroll_length = args.batch_size_total // args.n_envs
    steps_per_iteration = unroll_length * args.n_envs

    total_timesteps = args.n_iterations * steps_per_iteration
    
    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=logging_dir)
        def progress(num_steps, metrics):
            for key, value in metrics.items():
                writer.add_scalar(f"train/{key}", value, num_steps)
            print(f"Steps: {num_steps}, Reward: {metrics['eval/episode_reward']:.2f}")
    except ImportError:
        def progress(num_steps, metrics):
            print(f"Steps: {num_steps}, Reward: {metrics['eval/episode_reward']:.2f}")

    def policy_params_fn(current_step, make_policy, params):
        ckpt_dir = os.path.join(models_dir, f"{args.model_name}_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"step_{current_step}")
        brax_model.save_params(ckpt_path, params)

    # Bypass `jax.device_put_replicated` if removed
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import numpy as np
    from brax.envs import base as brax_env

    def _device_put_replicated_mock(x, devices):
        mesh = Mesh(np.array(devices), ('x',))
        sharding = NamedSharding(mesh, P('x'))
        return jax.tree.map(
            lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding), x
        )
    
    if not hasattr(jax, 'device_put_replicated'):
        jax.device_put_replicated = _device_put_replicated_mock

    env = build_environment(
        flat=args.flat,
        randomize_domain=args.dr,
        slippery_eval=False,
        init_k=args.init_k,
        exponent=args.exponent,
        seed=args.seed,
        num_envs=args.n_envs,
        reward_schedule_steps_per_iteration=steps_per_iteration,
    )
    env = TeacherObservationWrapper(env)

    eval_env = build_environment(
        flat=args.flat,
        randomize_domain=args.dr,
        slippery_eval=False,
        init_k=1.0,
        exponent=1.0,
        seed=args.seed,
        num_envs=128,
        reward_schedule_steps_per_iteration=None,
    )
    eval_env = TeacherObservationWrapper(eval_env)

    ppo_networks_factory = lambda obs_size, act_size, **kwargs: make_teacher_ppo_networks(
        obs_size, act_size, proprio_size=48, latent_size=8
    )

    make_inference_fn, params, metrics = ppo.train(
        environment=env,
        eval_env=eval_env,
        num_timesteps=args.n_iterations * steps_per_iteration,
        num_evals=args.n_iterations + 1,
        episode_length=1000,
        num_envs=args.n_envs,
        unroll_length=unroll_length,
        batch_size=(unroll_length * args.n_envs) // args.n_minibatches,
        num_minibatches=args.n_minibatches,
        num_updates_per_batch=args.n_epochs,
        learning_rate=args.learning_rate,
        entropy_cost=args.ent_coef,
        discounting=args.gamma,
        seed=args.seed,
        reward_scaling=1.0,
        randomization_fn=None,
        wrap_env=False,
        progress_fn=progress,
        network_factory=ppo_networks_factory,
        policy_params_fn=policy_params_fn,
    )

    model_path = os.path.join(models_dir, args.model_name)
    brax_model.save_params(model_path, params)
    print(f"Training finished. Model saved to {model_path}")

if __name__ == "__main__":
    main()
