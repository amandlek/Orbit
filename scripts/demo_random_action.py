"""
Script that experiments with controller action space.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Franka-v0", help="Name of the task.")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym
import torch
import numpy as np

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
import omni.isaac.orbit.utils.math as MathUtils
from omni.isaac.orbit_envs.utils import parse_env_cfg


def test_action_cycle_consistency(env, env_cfg, actions):
    """
    Small test to see if we can predict controller targets and then go back to actions.
    """

    # TODO: do we need to enforce quat convention (e.g. positive real part)
    source_pos = env.robot.data.ee_state_w[:, :3]
    source_rot = env.robot.data.ee_state_w[:, 3:7]

    # TODO: how to make new tensor so that changes are safe?

    # convert normalized actions to raw actions
    delta_pose = actions[:, :6].clone()
    delta_pose[:, :3] = delta_pose[:, :3] @ env.unwrapped._ik_controller._position_command_scale
    delta_pose[:, 3:6] = delta_pose[:, 3:6] @ env.unwrapped._ik_controller._rotation_command_scale
    target_pos, target_rot = MathUtils.apply_delta_pose(
            source_pos=source_pos,
            source_rot=source_rot,
            delta_pose=delta_pose,
    )
    actions_est = MathUtils.apply_delta_pose_inverse(
        source_pos=source_pos,
        source_rot=source_rot,
        target_pos=target_pos,
        target_rot=target_rot,
    )
    assert actions.shape[0] == 1
    assert actions_est.shape[0] == 1
    actions = actions.cpu().numpy()[0, :6]
    actions_est = actions_est.cpu().numpy()[0]

    # normalize actions
    pos_scale = env_cfg.control.inverse_kinematics.position_command_scale
    rot_scale = env_cfg.control.inverse_kinematics.rotation_command_scale
    actions_est[:3] = np.clip(actions_est[:3] / pos_scale, -1., 1.)
    actions_est[3:6] = np.clip(actions_est[3:6] / rot_scale, -1., 1.)
    print("action error: {}".format(np.abs(actions - actions_est)))
    print("controller_target_pos_est: {}".format(target_pos))
    print("controller_target_quat_est: {}".format(target_rot))


def main():
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=1)
    # modify configuration
    env_cfg.control.control_type = "inverse_kinematics"
    env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    env_cfg.terminations.episode_timeout = False
    env_cfg.terminations.is_success = True
    env_cfg.observations.return_dict_obs_in_group = True

    # TODO: check observations

    # position and rotation control scales
    pos_scale = env_cfg.control.inverse_kinematics.position_command_scale
    rot_scale = env_cfg.control.inverse_kinematics.rotation_command_scale

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)

    # robot pose
    eef_pos = env.robot.data.ee_state_w[:, :3]
    eef_quat_wxyz = env.robot.data.ee_state_w[:, 3:7]
    eef_quat_xyzw = MathUtils.convert_quat(eef_quat_wxyz, to="xyzw")

    # IK controller (see https://github.com/NVIDIA-Omniverse/Orbit/blob/main/source/extensions/omni.isaac.orbit/omni/isaac/orbit/controllers/differential_inverse_kinematics.py)
    ik_controller = env.unwrapped._ik_controller

    # number of envs
    num_envs = env.num_envs
    assert num_envs == 1

    # action space dimension
    ac_dim = env.action_space.shape[0]

    # reset environment
    obs_dict = env.reset()
    # robomimic only cares about policy observations
    obs = obs_dict["policy"]

    # simulate environment
    while simulation_app.is_running():
        # sample actions from -1 to 1
        actions = 2 * torch.rand((env.num_envs, env.action_space.shape[0]), device=env.device) - 1
        # print(actions)

        # predict controller targets, and cycle consistency with actions
        test_action_cycle_consistency(env=env, env_cfg=env_cfg, actions=actions)

        # apply actions
        obs_dict, _, _, _ = env.step(actions)

        # check if simulator is stopped
        if env.unwrapped.sim.is_stopped():
            break
        # robomimic only cares about policy observations
        obs = obs_dict["policy"]

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
