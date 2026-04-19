import jax
import jax.numpy as jp
from typing import Dict, Any, Tuple
from mujoco_playground import locomotion
from mujoco_playground._src import wrapper as mjx_wrapper
from brax.envs import base as brax_env
from mujoco import mjx

class RewardScheduleWrapper(brax_env.Wrapper):
    """
    Wraps the Brax environment OUTSIDE the vectorization/auto-reset boundaries to tracking the true global step
    and apply the curriculum `k` dynamically without wiping the counter at each episode reset.
    """
    def __init__(self, env, init_k: float, exponent: float, steps_per_iteration: int | None):
        super().__init__(env)
        self.init_k = float(init_k)
        self.exponent = float(exponent)
        self.steps_per_iteration = steps_per_iteration

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        info = state.info
        info['curriculum_global_step'] = jp.zeros((), dtype=jp.int32)
        
        metrics = state.metrics
        metrics['reward/rma_k'] = jp.asarray(self.init_k, dtype=jp.float32)
        return state.replace(info=info, metrics=metrics)

    def step(self, state, action):
        next_state = self.env.step(state, action)
        
        global_step = state.info['curriculum_global_step'] + 1
        
        if self.steps_per_iteration is None:
            k = jp.asarray(self.init_k, dtype=jp.float32)
        else:
            num_envs = next_state.reward.shape[0] if len(next_state.reward.shape) > 0 else 1
            total_transitions = global_step * num_envs
            iteration = (total_transitions // self.steps_per_iteration).astype(jp.float32)
            k = jp.power(self.init_k, jp.power(self.exponent, iteration))

        components = next_state.info['rma_components']
        
        if len(components.shape) > 1:
            scheduled_components = jp.concatenate([
                components[:, :2],
                components[:, 2:] * k,
            ], axis=1)
            scheduled_reward = jp.sum(scheduled_components, axis=1)
            
            info = next_state.info
            info['curriculum_global_step'] = global_step
            info['reward_k'] = jp.broadcast_to(k, scheduled_reward.shape)
            
            metrics = next_state.metrics
            for idx in range(10):
                metrics[f'reward/rma_component_{idx + 1}'] = jp.mean(components[:, idx])
                metrics[f'reward/rma_component_{idx + 1}_scheduled'] = jp.mean(scheduled_components[:, idx])
            metrics['reward/rma_total'] = jp.mean(scheduled_reward)
            metrics['reward/rma_k'] = k
        else:
            scheduled_components = jp.concatenate([
                components[:2],
                components[2:] * k,
            ])
            scheduled_reward = jp.sum(scheduled_components)
            
            info = next_state.info
            info['curriculum_global_step'] = global_step
            info['reward_k'] = k
            
            metrics = next_state.metrics
            for idx in range(10):
                metrics[f'reward/rma_component_{idx + 1}'] = components[idx]
                metrics[f'reward/rma_component_{idx + 1}_scheduled'] = scheduled_components[idx]
            metrics['reward/rma_total'] = scheduled_reward
            metrics['reward/rma_k'] = k

        return next_state.replace(reward=scheduled_reward, info=info, metrics=metrics)

def build_environment(
        flat: bool = False,
        randomize_domain: bool = False,
        slippery_eval: bool = False,
        init_k: float = 0.03,
        exponent: float = 0.997,
        seed: int = 42,
        num_envs: int = 4096,
        randomization_params: Dict[str, Any] | None = None,
        reward_schedule_steps_per_iteration: int | None = None,
        **kwargs
):
    env_name = "Go1JoystickFlatTerrain" if flat else "Go1JoystickRoughTerrain"
    
    config_overrides = {}
    
    env = locomotion.load(env_name, config_overrides=config_overrides)
    
    randomization_fn = None
    if randomize_domain:
        params = _default_randomization_params()
        if randomization_params is not None:
            params.update(randomization_params)
        if slippery_eval:
            params["friction"] = (0.1, 0.1)

        rng = jax.random.split(jax.random.PRNGKey(seed), num_envs)

        trunk_id = env.mj_model.body("trunk").id
        floor_geom_id = env.mj_model.geom("floor").id
        randomization_fn = _build_dr_function(
            model_template=env.mjx_model,
            trunk_id=trunk_id,
            floor_geom_id=floor_geom_id,
            rng=rng,
            randomization_params=params,
        )
            
    env = RMAWrapper(env, init_k=init_k)
    
    env = mjx_wrapper.wrap_for_brax_training(
        env,
        episode_length=1000,
        action_repeat=1,
        randomization_fn=randomization_fn,
        full_reset=False
    )
    
    env = RewardScheduleWrapper(
        env,
        init_k=init_k,
        exponent=exponent,
        steps_per_iteration=reward_schedule_steps_per_iteration
    )
    
    return env

def _default_randomization_params() -> Dict[str, Any]: 
    # From RMA paper Adjusted to Go1 
    return {
        "friction": (0.05, 4.5),
        "Kp": (0.9090909, 1.0909091),
        "Kd": (0.6666667, 1.3333333),
        "payload": (0.0, 6.0),
        "com": (-0.15, 0.15),
        "motor_strength": (0.90, 1.10),
        "resample_probability": 0.004,
    }


def _build_dr_function(
    model_template: mjx.Model,
    trunk_id: int,
    floor_geom_id: int,
    rng: jax.Array,
    randomization_params: Dict[str, Any],
):
    base_floor_friction = model_template.geom_friction[floor_geom_id, 0]
    base_kp = model_template.actuator_gainprm[:, 0]
    base_kd = model_template.dof_damping[6:]
    base_motor_strength = model_template.actuator_forcerange.copy()
    base_mass = model_template.body_mass
    base_com = model_template.body_ipos[trunk_id]

    resample_probability = float(randomization_params.get("resample_probability", 1.0))
    resample_probability = max(0.0, min(1.0, resample_probability))

    def randomizer(model: mjx.Model):
        @jax.vmap
        def randomize_one(single_rng):
            geom_friction = model.geom_friction
            actuator_gainprm = model.actuator_gainprm
            actuator_biasprm = model.actuator_biasprm
            dof_damping = model.dof_damping
            actuator_forcerange = model.actuator_forcerange
            body_mass = model.body_mass
            body_ipos = model.body_ipos

            single_rng, apply_key = jax.random.split(single_rng)
            apply_randomization = jax.random.uniform(apply_key) < resample_probability

            if "friction" in randomization_params:
                low, high = randomization_params["friction"]
                single_rng, key = jax.random.split(single_rng)
                friction_scale = jax.random.uniform(key, minval=low, maxval=high)
                randomized_friction = geom_friction.at[floor_geom_id, 0].set(base_floor_friction * friction_scale)
                geom_friction = jax.lax.select(apply_randomization, randomized_friction, geom_friction)

            if "Kp" in randomization_params:
                low, high = randomization_params["Kp"]
                single_rng, key = jax.random.split(single_rng)
                kp_scale = jax.random.uniform(key, shape=base_kp.shape, minval=low, maxval=high)
                randomized_kp = base_kp * kp_scale
                gain_candidate = actuator_gainprm.at[:, 0].set(randomized_kp)
                bias_candidate = actuator_biasprm.at[:, 1].set(-randomized_kp)
                actuator_gainprm = jax.lax.select(apply_randomization, gain_candidate, actuator_gainprm)
                actuator_biasprm = jax.lax.select(apply_randomization, bias_candidate, actuator_biasprm)

            if "Kd" in randomization_params:
                low, high = randomization_params["Kd"]
                single_rng, key = jax.random.split(single_rng)
                kd_scale = jax.random.uniform(key, shape=base_kd.shape, minval=low, maxval=high)
                randomized_kd = base_kd * kd_scale
                damp_candidate = dof_damping.at[6:].set(randomized_kd)
                dof_damping = jax.lax.select(apply_randomization, damp_candidate, dof_damping)

            if "payload" in randomization_params:
                low, high = randomization_params["payload"]
                single_rng, key = jax.random.split(single_rng)
                payload_value = jax.random.uniform(key, minval=low, maxval=high)
                candidate_mass = body_mass.at[trunk_id].set(base_mass[trunk_id] + payload_value)
                body_mass = jax.lax.select(apply_randomization, candidate_mass, body_mass)

            if "com" in randomization_params:
                low, high = randomization_params["com"]
                single_rng, key = jax.random.split(single_rng)
                com_low = low / 100.0
                com_high = high / 100.0
                delta_com_xy = jax.random.uniform(key, shape=(2,), minval=com_low, maxval=com_high)
                com_candidate = base_com.at[:2].set(base_com[:2] + delta_com_xy)
                candidate_ipos = body_ipos.at[trunk_id].set(com_candidate)
                body_ipos = jax.lax.select(apply_randomization, candidate_ipos, body_ipos)

            if "motor_strength" in randomization_params:
                low, high = randomization_params["motor_strength"]
                single_rng, key = jax.random.split(single_rng)
                strength_scale = jax.random.uniform(key, shape=(base_motor_strength.shape[0], 1), minval=low, maxval=high)
                candidate_forcerange = base_motor_strength * strength_scale
                actuator_forcerange = jax.lax.select(apply_randomization, candidate_forcerange, actuator_forcerange)

            return geom_friction, actuator_gainprm, actuator_biasprm, dof_damping, actuator_forcerange, body_mass, body_ipos

        (
            geom_friction,
            actuator_gainprm,
            actuator_biasprm,
            dof_damping,
            actuator_forcerange,
            body_mass,
            body_ipos,
        ) = randomize_one(rng)

        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace({
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
            "dof_damping": 0,
            "actuator_forcerange": 0,
            "body_mass": 0,
            "body_ipos": 0,
        })

        randomized_model = model.tree_replace({
            "geom_friction": geom_friction,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
            "dof_damping": dof_damping,
            "actuator_forcerange": actuator_forcerange,
            "body_mass": body_mass,
            "body_ipos": body_ipos,
        })

        return randomized_model, in_axes

    return randomizer


def _compute_reward_components(
    action,
    root_lin_vel,
    root_ang_vel,
    joint_ang,
    joint_vel,
    cur_torque,
    grnd_cfrc_ext,
    grav_vec,
    feet_vel,
    last_joint_ang,
    last_grnd_cfrc_ext,
    last_torque,
    target_reward_scaling,
):
    forward_reward = jp.minimum(root_lin_vel[0], 0.35) * target_reward_scaling[0]

    lat_mvmt_rot_penalty = -1.0 * (root_lin_vel[1] ** 2 + root_ang_vel[2] ** 2) * target_reward_scaling[1]

    work_penalty = -1.0 * jp.abs(jp.dot(cur_torque, (joint_ang - last_joint_ang))) * target_reward_scaling[2]

    ground_impact_penalty = -1.0 * jp.sum((grnd_cfrc_ext - last_grnd_cfrc_ext) ** 2) * target_reward_scaling[3]

    smoothness_penalty = -1.0 * jp.sum((cur_torque - last_torque) ** 2) * target_reward_scaling[4]

    action_magnitude_penalty = -1.0 * jp.sum(action ** 2) * target_reward_scaling[5]

    joint_speed_penalty = -1.0 * jp.sum(joint_vel ** 2) * target_reward_scaling[6]

    orientation_penalty = -1.0 * jp.sum(grav_vec[:2] ** 2) * target_reward_scaling[7]

    z_acc_penalty = -1.0 * root_lin_vel[2] ** 2 * target_reward_scaling[8]

    contact_mask = (grnd_cfrc_ext > 0.0).astype(feet_vel.dtype)
    foot_slip_penalty = -1.0 * jp.sum((contact_mask[:, None] * feet_vel) ** 2) * target_reward_scaling[9]

    return jp.array([
        forward_reward,
        lat_mvmt_rot_penalty,
        work_penalty,
        ground_impact_penalty,
        smoothness_penalty,
        action_magnitude_penalty,
        joint_speed_penalty,
        orientation_penalty,
        z_acc_penalty,
        foot_slip_penalty,
    ])

class RMAWrapper(mjx_wrapper.Wrapper):
    """
    Wraps the mujoco_playground environment to:
    1. Extract 'state' as the main observation for Brax PPO.
    2. Store 'privileged_state' in info for teacher training.
    """
    def __init__(self, env, init_k=0.03, **kwargs):
        super().__init__(env)
        self.init_k = float(init_k)

        self.target_reward_scaling = jp.array([
            20.0, 21.0, 0.002, 0.02, 0.001, 0.07, 0.002, 1.5, 2.0, 0.8
        ])
        self._foot_force_sensor_adr = jp.array([
            self.mj_model.sensor_adr[sensor_id] for sensor_id in self.env._feet_floor_found_sensor
        ])
        
    @property
    def observation_size(self):
        return self.env.observation_size["state"]

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        proprioceptive_obs = state.obs["state"]
        privileged_obs = state.obs["privileged_state"]
            
        info = state.info
        info['privileged_state'] = privileged_obs
        info['rma_components'] = jp.zeros(10, dtype=jp.float32)
        info['rma_last_joint_ang'] = state.data.qpos[7:]
        info['rma_last_torque'] = state.data.actuator_force
        info['rma_last_grnd_cfrc_ext'] = state.data.sensordata[self._foot_force_sensor_adr]
        
        return state.replace(obs=proprioceptive_obs, reward=jp.zeros(()), info=info)

    def step(self, state, action):
        last_joint_ang = state.info['rma_last_joint_ang']
        last_torque = state.info['rma_last_torque']
        last_grnd_cfrc_ext = state.info['rma_last_grnd_cfrc_ext']
        
        next_state = self.env.step(state, action)

        joint_ang = next_state.data.qpos[7:]
        joint_vel = next_state.data.qvel[6:]
        root_lin_vel = self.env.get_local_linvel(next_state.data)
        root_ang_vel = self.env.get_global_angvel(next_state.data)
        cur_torque = next_state.data.actuator_force
        grav_vec = self.env.get_gravity(next_state.data)
        feet_vel = next_state.data.sensordata[self.env._foot_linvel_sensor_adr]
        grnd_cfrc_ext = next_state.data.sensordata[self._foot_force_sensor_adr]

        reward_components = _compute_reward_components(
            action=action,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_ang=joint_ang,
            joint_vel=joint_vel,
            cur_torque=cur_torque,
            grnd_cfrc_ext=grnd_cfrc_ext,
            grav_vec=grav_vec,
            feet_vel=feet_vel,
            last_joint_ang=last_joint_ang,
            last_grnd_cfrc_ext=last_grnd_cfrc_ext,
            last_torque=last_torque,
            target_reward_scaling=self.target_reward_scaling,
        )

        proprioceptive_obs = next_state.obs["state"]
        privileged_obs = next_state.obs["privileged_state"]

        info = next_state.info
        info['privileged_state'] = privileged_obs
        info['rma_last_joint_ang'] = joint_ang
        info['rma_last_torque'] = cur_torque
        info['rma_last_grnd_cfrc_ext'] = grnd_cfrc_ext
        info['rma_components'] = reward_components

        return next_state.replace(obs=proprioceptive_obs, reward=jp.zeros(()), info=info)
