"""
environment.py : reward and DR wrapper for the environment based 
on the RMA paper : https://arxiv.org/abs/2107.04034
"""
from __future__ import annotations
import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from copy import copy
from typing import Callable

class reward_scheduler:
    def __init__(self, init_k=0.03, exponent=0.997, scaling_function : Callable =None):
        """
        Customise and schedule rewards. k will increase with k_t+1 = k_t ** exponent.
        scaling function can be customised, None is default RMA scaling
        with rewards[2:] (3 to 10) being scaled by a factor of k.
        return a single double as a reward.
        
        :param init_k: initial k value. (k0), supply 0 < k < 1 
        :param exponent: value to increase k by. use 0 < exponent < 1 
        :param scaling_function: given scaled rewards vector and k return a single number.
        :type scaling_function:
        """
        self.k = init_k
        self.exponent = exponent
        self.scaler = scaling_function if scaling_function is not None else self.rma_scaler
    
    def __call__(self, reward):
        return self.scaler(reward, self.k)

    def step_schedule(self):
        self.k = self.k ** self.exponent 

    @staticmethod
    def rma_scaler(reward, k):
        reward[2:] *= k
        return np.sum(reward)

    
def build_environment(
        flat : bool = False,
        reward_scheduling : reward_scheduler=None,
        randomize_domain : bool=False, 
        randomization_params : dict=None, 
        seed : int=42, 
        render_mode="rgb_array"
):
    
    xml_file = f"./mujoco_menagerie/unitree_go1{"_fractal" if not flat else ""}/scene.xml"

    if reward_scheduling is None:
        reward_scheduling = reward_scheduler(
            scaling_function=lambda reward, *args : np.sum(reward)
        )

    #taken from https://gymnasium.farama.org/tutorials/gymnasium_basics/load_quadruped_model/
    env = gym.make(
        "Ant-v5",
        xml_file=xml_file,
        forward_reward_weight=1,
        ctrl_cost_weight=0.05,
        contact_cost_weight=5e-4,
        healthy_reward=1,
        main_body=1,
        healthy_z_range=(0.195, 0.75),
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=0.1,
        frame_skip=5,
        max_episode_steps=1000,
        render_mode=render_mode,  # Change to "human" to visualize
        camera_name="tracking"
    )

    env = RMAEnv(
        env, 
        reward_scheduling=reward_scheduling,
        randomize_domain=randomize_domain, 
        randomization_parameters=randomization_params, 
        seed=seed
    )

    return env

class RMAEnv(Wrapper):
    def __init__(self, 
                 env : gym.Env,
                 reward_scheduling : reward_scheduler,
                 randomize_domain : bool = False,
                 randomization_parameters : dict = None,
                 seed : int = 42,
    ):
        """
        outputs observation state R30
        takes in action vector of target joint positions $a_t \in \mathbb{R}^12$ 

        rewards are given as a tuple of 10
        
        :param self: Description
        :param env: Description
        """
        super().__init__(env)

        #normalize action over standing position
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32, seed=seed
        )

        #nominal standing pose (Abduction, Hip, Knee) * 4
        self.nominal_qpos = np.array([0.0, 0.9, -1.8] * 4)
        #max deviation (radian)
        self.action_scale = 0.25

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(30,),
            dtype=np.float64,
            seed=seed,
        )


        self.target_reward_scaling = np.array([
            20, 21, 0.002, 0.02, 0.001, 0.07, 0.002, 1.5, 2.0, 0.8
        ]) #from RMA paper

        foot_names = [
            "FR_calf", "FL_calf", "RR_calf", "RL_calf"
        ]

        #correspond to mujoco env(0 is ground, 1 is trunk)
        self.foot_ids = [
            env.unwrapped.model.body(name).id for name in foot_names
        ]
        self.trunk_id = env.unwrapped.model.body("trunk").id 

        self._obs_hist = {
            'last_joint_ang' : np.zeros(12),
            'last_torque' : np.zeros(12),
            'last_grnd_cfrc_ext' : np.zeros((4, 6))
        }

        self._env_params = {
            'friction': copy(env.unwrapped.model.geom_friction[:, 0]),
            'Kp': copy(env.unwrapped.model.actuator_gainprm[:, 0]),
            'Kd': copy(env.unwrapped.model.dof_damping[6:]),
            'mass': copy(env.unwrapped.model.body_mass),
            'com': copy(env.unwrapped.model.body_ipos[self.trunk_id])
        }

        self.reward_scheduling = reward_scheduling
        self.randomize_domain = randomize_domain
        self.randomization_parameters = randomization_parameters

    def step(self, action):
        """
        - recieve normalized action vector in [-1, 1] and convert it to target joint positions.
        - reward is calculated as per RMA paper, then aggregated as per reward scheduler
        - domain randomization can be done for friction, Kp, Kd, payload, com and resample_probability.
        """
        if self.randomize_domain and self.randomization_parameters.get('resample_probability', None) is not None:
            if np.random.random() < self.randomization_parameters['resample_probability']:
                self._apply_domain_randomization()

        target_qpos = self.nominal_qpos + (action * self.action_scale)

        obs, old_reward, terminated, truncated, info = self.env.step(target_qpos)

        # slicing observation for proprioception
        # qpos
        root_pos = obs[:3]
        root_quat = obs[3:7]
        joint_ang = obs[7:19]
        # qvel 
        root_lin_vel = obs[19:22]
        root_ang_vel = obs[22:25]
        joint_vel = obs[25:37]
        # crfc_ext 13 bodies * 6D vector (3 torque, 3 linear)
        cfrc_ext = obs[37:]

        grnd_cfrc_ext = copy(self.env.unwrapped.data.cfrc_ext[self.foot_ids])
        # read torque from env
        cur_torque = np.copy(self.env.unwrapped.data.actuator_force)
        # calculate gravity vector of the trunk
        grav_vec = self._calculate_gravity_vector(root_quat)
        # calculate foot velocities
        feet_vel = copy(self.env.unwrapped.data.cvel[self.foot_ids][:, 3:])

        reward = self.calculate_reward(
            action,
            root_lin_vel,
            root_ang_vel,
            joint_ang,
            joint_vel,
            cur_torque,
            grnd_cfrc_ext,
            grav_vec,
            feet_vel
        )

        # RMA observations consist of 
        # (gravity_vec:3, root_ang_vel:3, joint_pos:12, joint_vel:12)
        obs = np.concat([
            grav_vec,
            root_ang_vel,
            joint_ang,
            joint_vel
        ])

        info.update({
            'root_pos' : root_pos,
            'root_quat': root_quat,
            'root_lin_vel' : root_lin_vel,
            'cfrc_ext' : cfrc_ext,
            'torque' : cur_torque,
            'feet_vel' : feet_vel,
        })

        self._obs_hist['last_joint_ang'] = joint_ang
        self._obs_hist['last_torque'] = cur_torque
        self._obs_hist['last_grnd_cfrc_ext'] = grnd_cfrc_ext        

        return obs, reward, terminated, truncated, info
        
        

    def reset(self, *, seed = None, options = None):

        #randomize domain
        if self.randomize_domain:
            self._apply_domain_randomization()

        #reset environment
        obs, info = super().reset(seed=seed, options=options)

        root_pos = obs[:3]
        root_quat = obs[3:7]
        joint_ang = obs[7:19]
        root_lin_vel = obs[19:22]
        root_ang_vel = obs[22:25]
        joint_vel = obs[25:37]
        cfrc_ext = obs[37:]

        # ground contact forces
        grnd_cfrc_ext = copy(self.env.unwrapped.data.cfrc_ext[self.foot_ids])
        # read torque
        cur_torque = np.copy(self.env.unwrapped.data.actuator_force)
        # calculate grav_vec
        grav_vec = self._calculate_gravity_vector(root_quat)
        # calculate foot velocities
        feet_vel = copy(self.env.unwrapped.data.cvel[self.foot_ids][:, 3:])

        obs = np.concat([
            grav_vec,       # 3
            root_ang_vel,   # 3
            joint_ang,      # 12
            joint_vel       # 12
        ],)

        info.update({
            'root_pos' : root_pos,
            'root_quat': root_quat,
            'root_lin_vel' : root_lin_vel,
            'cfrc_ext' : cfrc_ext,
            'torque' : cur_torque,
            'feet_vel' : feet_vel,
        })

        self._obs_hist['last_joint_ang'] = joint_ang
        self._obs_hist['last_torque'] = cur_torque
        self._obs_hist['last_grnd_cfrc_ext'] = grnd_cfrc_ext

        return obs, info
        
    def calculate_reward(
            self,
            action,
            root_lin_vel,
            root_ang_vel,
            joint_ang,
            joint_vel,
            cur_torque,
            grnd_cfrc_ext,
            grav_vec,
            feet_vel,
    ) -> float:
        #1
        forward_reward = (
            min(root_lin_vel[0], 0.35)    
        ) 
        #2
        lat_mvmt_rot_penalty = (
            -1 
            * (
                root_lin_vel[1]**2 
                + root_ang_vel[2]**2
            )
        )
        #3
        work_penalty = (
            -1
            * np.abs(np.dot(
                cur_torque, 
                (joint_ang - self._obs_hist['last_joint_ang'])
            ))
        )
        #4
        ground_impact_penalty = (
            -1
            *  np.linalg.norm(
                grnd_cfrc_ext - self._obs_hist['last_grnd_cfrc_ext']
            ) ** 2
        ) 
        #5
        smoothness_penalty = (
            -1
            * np.linalg.norm(
                cur_torque - self._obs_hist['last_torque']
            ) ** 2
        )
        #6
        action_magnitude_penalty = (
            -1
            * np.linalg.norm(action)**2
        )
        #7
        joint_speed_penalty = (
            -1
            * np.linalg.norm(joint_vel) ** 2
        )
        #8
        orientation_penalty = (
            -1
            * np.sum(grav_vec[:2]**2, axis=-1)
        )
        #9
        z_acc_penalty = (
            -1
            * root_lin_vel[2] ** 2
        )
        #10
        g = (np.linalg.norm(grnd_cfrc_ext[:, 3:], axis=-1) > 0).astype(np.float64)
        foot_slip_penalty = (
            -1
            * np.linalg.norm(
                np.diag(g) @ feet_vel #(4, 4) x (4, 3)
            ) ** 2
        )
        
        reward = self.target_reward_scaling * np.array((
            forward_reward,
            lat_mvmt_rot_penalty,
            work_penalty,
            ground_impact_penalty,
            smoothness_penalty,
            action_magnitude_penalty,
            joint_speed_penalty,
            orientation_penalty,
            z_acc_penalty,
            foot_slip_penalty
        ))

        return self.reward_scheduling(reward)
    

    def _apply_domain_randomization(self):
        model = self.env.unwrapped.model

        if 'friction' in self.randomization_parameters:
            low, high = self.randomization_parameters['friction']
            model.geom_friction[:, 0] = self._env_params['friction'] * np.random.uniform(low, high)

        if 'Kp' in self.randomization_parameters:
            low, high = self.randomization_parameters['Kp']
            new_gainprm = self._env_params['Kp'] * np.random.uniform(low, high, size=12)
            model.actuator_gainprm[:, 0] = new_gainprm

        if 'Kd' in self.randomization_parameters:
            low, high = self.randomization_parameters['Kd']
            model.dof_damping[6:] = self._env_params['Kd'] * np.random.uniform(low, high, size=12)

        if 'payload' in self.randomization_parameters:
            low, high = self.randomization_parameters['payload']
            new_body_mass = copy(self._env_params['mass'])
            new_body_mass[self.trunk_id] += np.random.uniform(low, high)
            model.body_mass[:] = new_body_mass

        if 'com' in self.randomization_parameters:
            low, high = self.randomization_parameters['com']
            com = self._env_params['com'] + np.random.uniform(low, high, size=3)
            model.body_ipos[self.trunk_id] = com

    def set_reward_k(self, k: float):
        self.reward_scheduling.k = k

    @staticmethod
    def _calculate_gravity_vector(quaternion):
        # def H(quat1, quat2):
        #     """
        #     ref: https://en.wikipedia.org/wiki/Quaternion#Hamilton_product

        #     :param quat1: quaternion
        #     :param quat2: quaternion
        #     """
        #     a1, b1, c1, d1 = quat1
        #     a2, b2, c2, d2 = quat2

        #     e1 = a1*a2 - b1*b2 - c1*c2 - d1*d2
        #     e2 = a1*b2 + b1*a2 + c1*d2 - d1*c2
        #     e3 = a1*c2 - b1*d2 + c1*a2 + d1*b2
        #     e4 = a1*d2 + b1*c2 - c1*b2 + d1*a2

        #     return np.array([e1, e2, e3, e4])
        
        # w, x, y, z = quaternion
        # inv_quaternion = np.array([w, -x, -y, -z])
        # abs_gravity = np.array([0, 0, 0, -1])

        # grav_quat = H(H(inv_quaternion, abs_gravity), quaternion)
        # grav_vec = grav_quat[1:]
        # return grav_vec

        #shortcut
        w, x, y, z = quaternion
        gx = 2 * (w * y - x * z)
        gy = -2 * (x * w + y * z)
        gz = -(1 - 2 * (x**2 + y**2))
        return np.array([gx, gy, gz])
    


if __name__ == "__main__":
    from pprint import pprint
    env = build_environment(reward_scheduling=reward_scheduler())
    obs0, info = env.reset()
    action = env.action_space.sample()
    obs1, reward, _, _, info = env.step(action)
    pprint(obs0)
    pprint(action)
    pprint(obs1)
    pprint(reward)
    pprint(f"obs space :  {env.observation_space}")
    
        

    