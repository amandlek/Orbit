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

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg


def main():
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=1)
    # modify configuration
    env_cfg.control.control_type = "inverse_kinematics"
    env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    env_cfg.terminations.episode_timeout = False
    env_cfg.terminations.is_success = True
    env_cfg.observations.return_dict_obs_in_group = True

    # TODO: take a look at the env cfg, and see the settings
    # TODO: ask about env cfg settings - documented somewhere?
    # TODO: how to get controller, and take a look at controller settings too?
    # TODO: check observations

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)

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
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[0])
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
