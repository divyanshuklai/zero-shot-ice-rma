"""
show_environment.py - Visualize the MJX environment using the MuJoCo passive viewer.
"""

import time
import jax
import mujoco
import mujoco.viewer
from mujoco_playground import locomotion

def main():
    print("Loading MJX environment...")
    env = locomotion.load("Go1JoystickFlatTerrain")
    
    rng = jax.random.PRNGKey(42)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(rng)

    m = env.mj_model
    d = mujoco.MjData(m)
    
    print("Launching MuJoCo viewer...")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            rng, action_rng = jax.random.split(rng)
            action = jax.random.uniform(action_rng, (env.action_size,), minval=-1.0, maxval=1.0)
            
            state = jit_step(state, action)
            mujoco.mjx.get_data_into(d, m, state.data)
            mujoco.mj_forward(m, d)
            
            if bool(state.done):
                print("Robot fell! Resetting...")
                rng, reset_rng = jax.random.split(rng)
                state = jit_reset(reset_rng)

            viewer.sync()

            time_until_next_step = env.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
