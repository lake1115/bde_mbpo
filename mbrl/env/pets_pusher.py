import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, render_mode: str = None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(
            self, "%s/assets/pusher.xml" % dir_path, 4, observation_space, render_mode
        )
        utils.EzPickle.__init__(self)
        self.reset_model()

    def step(self, a):
        obj_pos = (self.get_body_com("object"),)
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")

        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = -np.square(a).sum()
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        terminated = False

        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, False, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        self.ac_goal_pos = self.get_body_com("goal")

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:7],
                self.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
            ]
        )

    def _get_state(self):
        return np.concatenate(
            [
                self.data.qpos.flat,
                self.data.qvel.flat,
            ]
        )
