"""
Script that shows problems with first few timesteps of simulation.
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


def warmstart_env(env, num_steps=100, only_sim_step=False):
    """
    Issue a few zero actions to let the env settle, and return the most recent eef pose.
    """
    actions = 2 * torch.rand((env.num_envs, env.action_space.shape[0]), device=env.device) - 1
    actions = torch.zeros_like(actions)
    actions[:, -1] = -1.
    for _ in range(num_steps):
        if only_sim_step:
            env.sim.step()
        else:
            env.step(actions)

    return env.robot.data.ee_state_w[:, :3].clone(), env.robot.data.ee_state_w[:, 3:7].clone()


def run_trial(env, num_steps, only_sim_step):
    """
    Args:
        num_steps (int): number of warmstart steps
        only_sim_step (bool): if True, only call env.sim.step instead
            of env.step
    """
    # reset
    env.reset()

    # initial robot pose
    eef_pos = env.robot.data.ee_state_w[:, :3].clone()
    eef_quat_wxyz = env.robot.data.ee_state_w[:, 3:7].clone()

    # warmstart the environment to allow things to settle
    ws_eef_pos, ws_eef_quat = warmstart_env(env=env, num_steps=num_steps, only_sim_step=only_sim_step)

    # get error between warmstarted eef pose and first eef pose
    pos_err, quat_err = MathUtils.compute_pose_error(
        t01=eef_pos,
        q01=eef_quat_wxyz,
        t02=ws_eef_pos,
        q02=ws_eef_quat,
        rot_error_type="axis_angle",
    )
    print("eef pos 1: {}".format(eef_pos))
    print("eef pos 2: {}".format(ws_eef_pos))
    print("position error: {}".format(pos_err.squeeze()))
    quat_err_mag = quat_err.norm(dim=-1).squeeze()
    print("quat error magnitude: {} radians ({} degrees)".format(quat_err_mag, (180. / np.pi) * quat_err_mag))


def main():
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=1)
    # modify configuration
    env_cfg.control.control_type = "inverse_kinematics"
    env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    env_cfg.terminations.episode_timeout = False
    env_cfg.terminations.is_success = True
    env_cfg.observations.return_dict_obs_in_group = True

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)

    # number of envs
    num_envs = env.num_envs
    assert num_envs == 1

    # NOTE: warmstart of 1 step vs. 100 steps results in the same amount of error
    # NOTE: only using env.sim.step doesn't change the pose much
    print("\nrun trial")
    run_trial(env=env, num_steps=1, only_sim_step=False)
    # run_trial(env=env, num_steps=100, only_sim_step=False)
    # run_trial(env=env, num_steps=1, only_sim_step=True)
    # run_trial(env=env, num_steps=100, only_sim_step=True)

    print("\ntrying again")
    run_trial(env=env, num_steps=1, only_sim_step=False)

    # from IPython import embed; embed()

    # # simulate environment
    # while simulation_app.is_running():
    #     # sample actions from -1 to 1
    #     actions = 2 * torch.rand((env.num_envs, env.action_space.shape[0]), device=env.device) - 1
    #     # actions = torch.zeros_like(actions)
    #     # actions[:, -1] = -1.

    #     # apply actions
    #     print(actions)
    #     obs_dict, _, _, _ = env.step(actions)

    #     # check if simulator is stopped
    #     if env.unwrapped.sim.is_stopped():
    #         break
    #     # robomimic only cares about policy observations
    #     obs = obs_dict["policy"]

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
