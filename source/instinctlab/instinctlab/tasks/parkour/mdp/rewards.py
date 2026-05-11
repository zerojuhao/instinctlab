from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(env, command_name: str, vel_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # no reward for zero command
    reward *= torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward


# def stand_still(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     threshold: float = 0.15,
#     offset: float = 1.0,
# ) -> torch.Tensor:
#     """Penalize moving when there is no velocity command."""
#     asset = env.scene[asset_cfg.name]
#     dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
#     return (
#         (dof_error - offset)
#         * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < threshold)
#         * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < threshold)
#     )

def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_pos_weight: float = 0.15,
    joint_vel_weight: float = 0.05,
    body_vel_weight: float = 0.8,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene["robot"]
    cmd = (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) + torch.abs(env.command_manager.get_command(command_name)[:, 2])
    )
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    pos_reward = joint_pos_weight *torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    vel_reward = joint_vel_weight * torch.sum(torch.abs(asset.data.joint_vel), dim=1)
    body_vel_reward = body_vel_weight * body_vel
    reward = torch.where(
        cmd > 0.1,
        0.0,
        pos_reward + vel_reward + body_vel_reward,
    )
    return reward

def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def left_thigh_yaw_joint_sign_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_name: str = "left_thigh_yaw_joint",
) -> torch.Tensor:
    """Penalize left thigh yaw above zero."""
    asset = env.scene[asset_cfg.name]
    if joint_name not in asset.joint_names:
        return torch.zeros(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)
    joint_id = asset.joint_names.index(joint_name)
    return torch.clamp(asset.data.joint_pos[:, joint_id], min=0.0)


def right_thigh_yaw_joint_sign_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_name: str = "right_thigh_yaw_joint",
) -> torch.Tensor:
    """Penalize right thigh yaw below zero."""
    asset = env.scene[asset_cfg.name]
    if joint_name not in asset.joint_names:
        return torch.zeros(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)
    joint_id = asset.joint_names.index(joint_name)
    return torch.clamp(-asset.data.joint_pos[:, joint_id], min=0.0)


def body_distance_y(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min: float = 0.2, max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0.)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

def feet_close_xy_gauss(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    # Get feet positions (assuming first two body_ids are left and right feet)
    left_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    right_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :2]
    heading_w = asset.data.heading_w

    # Transform feet positions to robot frame
    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    # Rotate to robot frame
    left_foot_robot_frame = torch.stack(
        [
            cos_heading * left_foot_xy[:, 0] + sin_heading * left_foot_xy[:, 1],
            -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1],
        ],
        dim=1,
    )

    right_foot_robot_frame = torch.stack(
        [
            cos_heading * right_foot_xy[:, 0] + sin_heading * right_foot_xy[:, 1],
            -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1],
        ],
        dim=1,
    )

    feet_distance_y = torch.abs(left_foot_robot_frame[:, 1] - right_foot_robot_frame[:, 1])

    # Return continuous penalty using exponential decay
    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def heading_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the heading error between the robot's current heading and the goal heading."""
    # compute the error
    ang_vel_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return ang_vel_cmd


def dont_wait(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())


def feet_orientation_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    left_projected_gravity = quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    right_projected_gravity = quat_apply_inverse(right_quat, asset.data.GRAVITY_VEC_W)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1

    return (
        torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
        + torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 1]
    )


def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset=0.035,
) -> torch.Tensor:
    """Reward feet being at certain height above the ground plane."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1
    left_sensor = env.scene[left_height_scanner_cfg.name]
    left_sensor_data = left_sensor.data.ray_hits_w[..., 2]
    left_sensor_data = torch.where(torch.isinf(left_sensor_data), 0.0, left_sensor_data)
    right_sensor = env.scene[right_height_scanner_cfg.name]
    right_sensor_data = right_sensor.data.ray_hits_w[..., 2]
    right_sensor_data = torch.where(torch.isinf(right_sensor_data), 0.0, right_sensor_data)
    left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    left_reward = (
        torch.clamp(left_height.unsqueeze(-1) - left_sensor_data - height_offset, min=0.0, max=0.3) * is_contact[:, 0:1]
    )
    right_reward = (
        torch.clamp(right_height.unsqueeze(-1) - right_sensor_data - height_offset, min=0.0, max=0.3)
        * is_contact[:, 1:2]
    )
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.GRAVITY_VEC_W)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 2 * forces_z, dim=1).float()
    return reward

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 1.0) / 1.0
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 1.0) / 1.0
    return reward



"""
Actor MLP: MoeLayer(
  (act_fn): ELU(alpha=1.0)
  (gate): Sequential(
    (0): Linear(in_features=752, out_features=10, bias=True)
  )
  (experts): ModuleList(
    (0-9): 10 x Sequential(
      (0): Linear(in_features=752, out_features=256, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=256, out_features=128, bias=True)
      (3): ELU(alpha=1.0)
      (4): Linear(in_features=128, out_features=64, bias=True)
      (5): ELU(alpha=1.0)
      (6): Linear(in_features=64, out_features=23, bias=True)
    )
  )
)
Critic MLP: MoeLayer(
  (act_fn): ELU(alpha=1.0)
  (gate): Sequential(
    (0): Linear(in_features=1304, out_features=10, bias=True)
  )
  (experts): ModuleList(
    (0-9): 10 x Sequential(
      (0): Linear(in_features=1304, out_features=256, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=256, out_features=128, bias=True)
      (3): ELU(alpha=1.0)
      (4): Linear(in_features=128, out_features=64, bias=True)
      (5): ELU(alpha=1.0)
      (6): Linear(in_features=64, out_features=1, bias=True)
    )
  )
)
Actor Encoder: ParallelLayer(1 blocks): ModuleDict(
  (depth_encoder): Conv2dHeadModel(
    (conv): Conv2dModel(
      (conv): Sequential(
        (0): Conv2d(8, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
      )
    )
    (head): MlpModel(
      (model): Sequential(
        (0): Linear(in_features=2304, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=128, bias=True)
        (5): ReLU()
      )
    )
  )
)
Critic Encoder: ParallelLayer(1 blocks): ModuleDict(
  (depth_encoder): Conv2dHeadModel(
    (conv): Conv2dModel(
      (conv): Sequential(
        (0): Conv2d(8, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
      )
    )
    (head): MlpModel(
      (model): Sequential(
        (0): Linear(in_features=2304, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=128, bias=True)
        (5): ReLU()
      )
    )
  )
)
Discriminator Network Structure: Discriminator(
  (model): MlpModel(
    (model): Sequential(
      (0): Linear(in_features=550, out_features=1024, bias=True)
      (1): ReLU()
      (2): Linear(in_features=1024, out_features=512, bias=True)
      (3): ReLU()
      (4): Linear(in_features=512, out_features=1, bias=True)
    )
  )
)

+-------------------------------------------------+
|   Active Observation Terms in Group: 'policy'   |
+--------+--------------------+-------------------+
| Index  | Name               |       Shape       |
+--------+--------------------+-------------------+
|   0    | base_ang_vel       |  (np.int64(24),)  |
|   1    | projected_gravity  |  (np.int64(24),)  |
|   2    | velocity_commands  |  (np.int64(24),)  |
|   3    | joint_pos          |  (np.int64(184),) |
|   4    | joint_vel          |  (np.int64(184),) |
|   5    | actions            |  (np.int64(184),) |
|   6    | depth_image        |    (8, 18, 32)    |
+--------+--------------------+-------------------+
+-------------------------------------------------+
|   Active Observation Terms in Group: 'critic'   |
+--------+--------------------+-------------------+
| Index  | Name               |       Shape       |
+--------+--------------------+-------------------+
|   0    | base_lin_vel       |  (np.int64(24),)  |
|   1    | base_ang_vel       |  (np.int64(24),)  |
|   2    | projected_gravity  |  (np.int64(24),)  |
|   3    | velocity_commands  |  (np.int64(24),)  |
|   4    | joint_pos          |  (np.int64(184),) |
|   5    | joint_vel          |  (np.int64(184),) |
|   6    | actions            |  (np.int64(184),) |
|   7    | depth_image        |    (8, 18, 32)    |
|   8    | height_scan        |  (np.int64(528),) |
+--------+--------------------+-------------------+
+---------------------------------------------------+
|  Active Observation Terms in Group: 'amp_policy'  |
+--------+---------------------+--------------------+
| Index  | Name                |       Shape        |
+--------+---------------------+--------------------+
|   0    | projected_gravity   |  (np.int64(30),)   |
|   1    | joint_pos_rel       |  (np.int64(230),)  |
|   2    | joint_vel           |  (np.int64(230),)  |
|   3    | base_lin_vel        |  (np.int64(30),)   |
|   4    | base_ang_vel        |  (np.int64(30),)   |
+--------+---------------------+--------------------+
+-----------------------------------------------------+
|  Active Observation Terms in Group: 'amp_reference' |
+--------+----------------------+---------------------+
| Index  | Name                 |        Shape        |
+--------+----------------------+---------------------+
|   0    | projected_gravity    |   (np.int64(30),)   |
|   1    | joint_pos_rel        |   (np.int64(230),)  |
|   2    | joint_vel            |   (np.int64(230),)  |
|   3    | base_lin_vel         |   (np.int64(30),)   |
|   4    | base_ang_vel         |   (np.int64(30),)   |
+--------+----------------------+---------------------+

"""