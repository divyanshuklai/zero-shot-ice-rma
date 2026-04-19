"""
show_environment.py - Visualize the MJX environment using the MuJoCo passive viewer.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

import time
import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mujoco.viewer
from mujoco_playground import locomotion
from environment import build_environment

def main():
    print("Loading MJX environment...")
    env = build_environment(
        flat=False,
        slippery_eval=False,
        num_envs=1
    )
    
    rng = jax.random.PRNGKey(42)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    reset_rng = jax.random.split(rng, 1)
    state = jit_reset(reset_rng)

    m = env.mj_model
    d = mujoco.MjData(m)
    
    print("Launching MuJoCo viewer...")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            rng, action_rng = jax.random.split(rng)
            action = jax.random.uniform(action_rng, (1, env.action_size), minval=-1.0, maxval=1.0)
            
            state = jit_step(state, action)
            
            # Sync only the minimal kinematics to the CPU for visualization
            # This completely avoids complex PyTree unbatching over Warp-specific struct internals
            d.qpos[:] = np.asarray(state.data.qpos[0])
            d.qvel[:] = np.asarray(state.data.qvel[0])
            d.time = float(np.asarray(state.data.time[0]))
            
            mujoco.mj_forward(m, d)
            viewer.sync()

            time_until_next_step = env.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
