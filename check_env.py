
import jax
import jax.numpy as jp
from environment import build_environment

def main():
    env = build_environment(num_envs=1, randomize_domain=True)
    rng = jax.random.PRNGKey(0)
    reset_rng = jax.random.split(rng, 1)
    state = env.reset(reset_rng)
    
    print(f"Observation keys: {state.obs.keys()}")
    for k, v in state.obs.items():
        print(f"  {k} shape: {v.shape}")
    print(f"Privileged obs shape: {state.info['privileged_state'].shape}")
    print(f"Action size: {env.action_size}")
    
    # Try to see what's in privileged_state
    # print(f"Privileged obs sample: {state.info['privileged_state']}")

if __name__ == '__main__':
    main()
