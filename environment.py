"""
environment.py : reward and DR wrapper for the environment based 
on the RMA paper : https://arxiv.org/abs/2107.04034
"""
from __future__ import annotations
import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from copy import copy

class reward_scheduler:
    def __init__(self, init_k=0.03, exponent=0.997, scaling_function : function =None):
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
        reward = self.scaler(reward, self.k)
        self.k = self.k**self.exponent
        return reward

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
        outputs observation state $x_t \in \mathbb{R}^30$
        takes in action vector of target joint positions $a_t \in \mathbb{R}^12$ 

        rewards are given as a tuple of 10
        
        :param self: Description
        :param env: Description
        """
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(30,),
            dtype=np.float64,
            seed=seed,
        )


        self.target_reward_scaling = np.array([ #reward scaling based on the RMA paper
            20,     # 1.forward
            21,     # 2.lateral movement and rotation
            0.002,  # 3.work
            0.02,   # 4.ground impact
            0.001,  # 5.smoothness
            0.07,   # 6.action magnitude
            0.002,  # 7.joint speed
            1.5,    # 8.orientation
            2.0,    # 9.z acceleration
            0.8     # 10.foot slip
        ])

        foot_names = [
            "FR_calf", 
            "FL_calf", 
            "RR_calf", 
            "RL_calf"
        ]

        self.foot_ids = [env.unwrapped.model.body(name).id for name in foot_names]
        self.trunk_id = env.unwrapped.model.body("trunk").id 

        self._obs_hist = {
            'last_joint_ang' : np.zeros(env.action_space.shape),
            'last_torque' : np.zeros(env.action_space.shape),
            'last_grnd_cfrc_ext' : np.zeros((len(self.foot_ids), 6))
        }

        self.reward_scheduling = reward_scheduling

    def step(self, action):
        obs, old_reward, terminated, truncated, info = self.env.step(action)

        # slicing observation for proprioception
        # part 1: qpos
        root_pos = obs[:3]
        root_quat = obs[3:7]
        joint_ang = obs[7:19]
        # part 2: qvel 
        root_lin_vel = obs[19:22]
        root_ang_vel = obs[22:25]
        joint_vel = obs[25:37]
        # part 3: crfc_ext 13 bodies * 6D vector (3 torque, 3 linear)
        cfrc_ext = obs[37:]

        # ground contact forces
        grnd_cfrc_ext = copy(self.env.unwrapped.data.cfrc_ext[self.foot_ids])
        # calculate torque
        cur_torque = self.calculate_torque(action)
        # calculate grav_vec
        grav_vec = self.calculate_gravity_vector(root_quat)
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
        # calculate torque
        cur_torque = self.calculate_torque(np.zeros(self.env.action_space.shape))
        # calculate grav_vec
        grav_vec = self.calculate_gravity_vector(root_quat)
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
    ) -> tuple[float]:
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
            * np.sum(grav_vec[...,:2]**2, axis=-1)
        )
        #9
        z_acc_penalty = (
            -1
            * root_lin_vel[2] ** 2
        )
        #10
        g = (np.sum(grnd_cfrc_ext, axis=-1) > 0).astype(np.float64)
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
        
    def calculate_torque(self, qpost):
        """
        PD Controller
        
        :param self: Description
        :param qpost: Description
        """
        Kp = self.env.unwrapped.model.actuator_gainprm[:, 0]
        Kd = self.env.unwrapped.model.dof_damping[6:]
        qvel = self.env.unwrapped.data.qvel[6:]
        qvelt = 0.0
        qpos =  self.env.unwrapped.data.qpos[7:]
        return Kp * (qpost - qpos) + Kd * (qvelt - qvel)
    
    @staticmethod
    def calculate_gravity_vector(quaternion):
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
    
        

    