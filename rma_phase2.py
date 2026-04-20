"""
rma_phase2 --> training student adaptation module using trained teacher policy.
"""

import os
import json
import argparse
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from datetime import datetime
from typing import Dict, Any, Optional

from environment import build_environment
from brax.envs import base as brax_env
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from rma import TeacherEncoder, TeacherNetwork, StudentEncoder

class StudentObservationWrapper(brax_env.Wrapper):
    """
    Maintains a history of proprio observations.
    """
    def __init__(self, env, history_len=50):
        super().__init__(env)
        self.proprio_size = 48
        self.privileged_size = 123
        self.history_len = history_len

    @property
    def observation_size(self):
        return self.proprio_size + self.privileged_size + self.history_len * self.proprio_size

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        info = state.info.copy()
        
        # We must use a batched shape for was_dict, else Brax's vmap wrapper will crash
        # state.reward has shape [batch_size]
        batch_shape = state.reward.shape
        
        if isinstance(state.obs, dict):
            proprio = state.obs["state"]
            privileged = state.obs["privileged_state"]
            info["was_dict"] = jnp.full(batch_shape, True)
        else:
            proprio = state.obs[..., :self.proprio_size]
            privileged = state.info["privileged_state"]
            info["was_dict"] = jnp.full(batch_shape, False)
        
        # Initialize history
        history = jnp.repeat(proprio[..., None, :], self.history_len, axis=-2)
        flat_history = history.reshape(history.shape[:-2] + (-1,))
        obs = jnp.concatenate([proprio, privileged, flat_history], axis=-1)
        
        info["history"] = history
        return state.replace(obs=obs, info=info)

    def step(self, state, action):
        proprio = state.obs[..., :self.proprio_size]
        privileged = state.obs[..., self.proprio_size:self.proprio_size+self.privileged_size]
        
        # JAX requires the carry of scan to have the same pytree structure.
        # So we must reconstruct the original obs structure before calling inner step.
        was_dict = state.info.get("was_dict")
        
        def make_dict(p, pr):
            return {"state": p, "privileged_state": pr}
            
        def make_array(p, pr):
            return p
            
        # Since we can't use Python if/else easily with JAX tracing if was_dict is dynamic,
        # but was_dict is actually static across steps for a given env setup.
        # Actually, because it's a structural difference, JAX will trace both branches if we use lax.cond,
        # which requires same output structure. BUT the inner env structure is fixed per compilation!
        # So we can just check the python type of the inner env or rely on a python boolean.
        # Since we added was_dict as an array, we can't use it to change pytree structure.
        # Let's just check if "privileged_state" is in state.info.
        if "privileged_state" not in state.info:
            original_obs = {"state": proprio, "privileged_state": privileged}
        else:
            original_obs = proprio
            
        state = state.replace(obs=original_obs)
        state = self.env.step(state, action)
        
        if isinstance(state.obs, dict):
            new_proprio = state.obs["state"]
            new_privileged = state.obs["privileged_state"]
        else:
            new_proprio = state.obs[..., :self.proprio_size]
            new_privileged = state.info["privileged_state"]
        
        old_history = state.info["history"]
        history = jnp.concatenate([old_history[..., 1:, :], new_proprio[..., None, :]], axis=-2)
        flat_history = history.reshape(history.shape[:-2] + (-1,))
        
        obs = jnp.concatenate([new_proprio, new_privileged, flat_history], axis=-1)
        
        info = state.info.copy()
        info["history"] = history
        return state.replace(obs=obs, info=info)

DEFAULTS = dict(
    seed=42,
    n_iterations=1000,
    batch_size_total=80_000,
    n_minibatches=4,
    n_epochs=1,
    n_envs=4096, 
    learning_rate=5e-4,
    flat=False,
    dr=True,
    model_name="",
    teacher_model_path="models/teacher_model_quick",
    log_dir=".",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RMA Phase 2 training for the Student Encoder."
    )
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=v)
        else:
            parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    
    args = parser.parse_args()
    if not args.model_name:
        date_str = datetime.now().strftime("%d%m%y")
        args.model_name = f"rma_phase2_mjx_{date_str}"
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

    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=logging_dir)
    except ImportError:
        writer = None

    env = build_environment(
        flat=args.flat,
        randomize_domain=args.dr,
        slippery_eval=False,
        init_k=1.0,
        exponent=1.0,
        seed=args.seed,
        num_envs=args.n_envs,
        reward_schedule_steps_per_iteration=None,
    )
    env = StudentObservationWrapper(env, history_len=50)
    action_size = env.action_size
    
    teacher_encoder = TeacherEncoder(latent_size=8)
    teacher_network = TeacherNetwork(action_size=action_size)
    student_encoder = StudentEncoder(latent_size=8)
    
    # Load teacher params
    teacher_params = brax_model.load_params(args.teacher_model_path)
    policy_params = teacher_params[1]['params']
    teacher_enc_params = {'params': policy_params['TeacherEncoder_0']}
    teacher_net_params = {'params': policy_params['TeacherNetwork_0']}
    teacher_log_std = policy_params['log_std']
    
    key = jax.random.PRNGKey(args.seed)
    key, key_student, key_env = jax.random.split(key, 3)
    
    # Init student encoder
    dummy_history = jnp.zeros((1, 50, 48))
    student_params = student_encoder.init(key_student, dummy_history)
    
    optimizer = optax.adam(learning_rate=args.learning_rate)
    opt_state = optimizer.init(student_params)
    
    env_reset = jax.jit(env.reset)
    env_step = jax.jit(env.step)
    env_state = env_reset(jax.random.split(key_env, args.n_envs))
    
    unroll_length = args.batch_size_total // args.n_envs

    def rollout_step(carry, unused):
        state, student_params, key = carry
        key, key_action = jax.random.split(key)
        
        proprio = state.obs[..., :48]
        privileged = state.obs[..., 48:48+123]
        flat_history = state.obs[..., 48+123:]
        history = flat_history.reshape(flat_history.shape[:-1] + (50, 48))
        
        # Student predicts z_hat
        z_hat = student_encoder.apply(student_params, history)
        
        # Teacher generates action using z_hat
        logits, _ = teacher_network.apply(teacher_net_params, proprio, z_hat)
        
        # Sample action
        std = jnp.exp(teacher_log_std)
        action = jax.random.normal(key_action, logits.shape) * std + logits
        
        next_state = env_step(state, action)
        
        # True z from TeacherEncoder
        true_z = teacher_encoder.apply(teacher_enc_params, privileged)
        
        data = {"history": history, "true_z": true_z}
        return (next_state, student_params, key), data

    def loss_fn(student_params, batch):
        z_hat = student_encoder.apply(student_params, batch["history"])
        loss = jnp.mean(jnp.square(z_hat - batch["true_z"]))
        return loss

    @jax.jit
    def train_iteration(env_state, student_params, opt_state, key):
        # 1. Rollout to collect data
        key, key_rollout = jax.random.split(key)
        (env_state, _, key_rollout), data = jax.lax.scan(
            rollout_step, (env_state, student_params, key_rollout), (), length=unroll_length
        )
        
        # Flatten batch
        # data["history"] shape: [unroll_length, n_envs, 50, 48] -> [unroll_length * n_envs, 50, 48]
        batch_size = unroll_length * args.n_envs
        flat_data = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), data)
        
        # 2. SGD over minibatches
        minibatch_size = batch_size // args.n_minibatches
        
        def update_step(carry, i):
            student_params, opt_state, key_update = carry
            mb_history = jax.lax.dynamic_slice_in_dim(flat_data["history"], i * minibatch_size, minibatch_size)
            mb_true_z = jax.lax.dynamic_slice_in_dim(flat_data["true_z"], i * minibatch_size, minibatch_size)
            mb = {"history": mb_history, "true_z": mb_true_z}
            
            loss, grads = jax.value_and_grad(loss_fn)(student_params, mb)
            updates, opt_state = optimizer.update(grads, opt_state, student_params)
            student_params = optax.apply_updates(student_params, updates)
            
            return (student_params, opt_state, key_update), loss

        indices = jnp.arange(args.n_minibatches)
        (student_params, opt_state, key), losses = jax.lax.scan(
            update_step, (student_params, opt_state, key), indices
        )
        
        return env_state, student_params, opt_state, key, jnp.mean(losses)

    print(f"Starting Phase 2 training for {args.n_iterations} iterations...")
    for it in range(args.n_iterations):
        env_state, student_params, opt_state, key, loss = train_iteration(
            env_state, student_params, opt_state, key
        )
        if it % 10 == 0 or it == args.n_iterations - 1:
            print(f"Iteration {it}, MSE Loss: {loss:.6f}")
            if writer:
                writer.add_scalar("train/mse_loss", loss.item(), it)
            if it % 100 == 0 or it == args.n_iterations - 1:
                ckpt_dir = os.path.join(models_dir, f"{args.model_name}_checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"iter_{it}")
                brax_model.save_params(ckpt_path, student_params)
                
    model_path = os.path.join(models_dir, args.model_name)
    brax_model.save_params(model_path, student_params)
    print(f"Phase 2 Training finished. Student model saved to {model_path}")

if __name__ == "__main__":
    main()
