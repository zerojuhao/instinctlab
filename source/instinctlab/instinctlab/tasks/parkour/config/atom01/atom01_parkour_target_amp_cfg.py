import copy
import os

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

from instinctlab.assets.roboparty import ATOM01_CFG, ATOM01_LINKS, ATOM01_SYMMETRIC_AUGMENTATION_JOINT_MAPPING, ATOM01_SYMMETRIC_AUGMENTATION_JOINT_REVERSE_BUF
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.parkour.config.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg

ATOM01_CFG.init_state.pos = (0.0, 0.0, 0.85)
@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    path = os.path.expanduser("/home/zym/instinct_train/atom01_lab")
    retargetting_func = None
    filtered_motion_selection_filepath = os.path.expanduser("/home/zym/instinct_train/atom01_lab/atom01.yaml")
    motion_start_from_middle_range = [0.0, 0.9]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontward"


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    robot_model_path=ATOM01_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/base_link",
    symmetric_augmentation_link_mapping=[0, 2, 1, 4, 3, 6, 5],
    symmetric_augmentation_joint_mapping=ATOM01_SYMMETRIC_AUGMENTATION_JOINT_MAPPING,
    symmetric_augmentation_joint_reverse_buf=ATOM01_SYMMETRIC_AUGMENTATION_JOINT_REVERSE_BUF,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={
        "run_walk": AmassMotionCfg(),
    },
    link_of_interests=[
        # "base_link",
        "torso_link",
        # "left_arm_roll_link",
        # "right_arm_roll_link",
        "left_elbow_pitch_link",
        "right_elbow_pitch_link",
        # "left_elbow_yaw_link",
        # "right_elbow_yaw_link",
        # "left_thigh_roll_link",
        # "right_thigh_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    mp_split_method="Even",
)


ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class Atom01ParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(ATOM01_LINKS))
        self.scene.motion_reference = motion_reference_cfg


class ShoeConfigMixin:
    def apply_shoe_config(self):
        self.scene.robot = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.leg_volume_points.points_generator.z_min = -0.063
        self.scene.leg_volume_points.points_generator.z_max = -0.023
        self.rewards.rewards.feet_at_plane.params["height_offset"] = 0.058



@configclass
class Atom01ParkourRoughEnvCfg_PLAY(Atom01ParkourRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        self.scene.num_envs = 10
        # self.viewer = ViewerCfg(
        #     eye=[4.0, 0.75, 1.0],
        #     lookat=[0.0, 0.75, 0.0],
        #     origin_type="asset_root",
        #     asset_name="robot",
        # )

        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None

        self.commands.base_velocity.velocity_ranges["pyramid_stairs"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        self.commands.base_velocity.velocity_ranges["pyramid_stairs_high"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv_high"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["hf_pyramid_slope_inv"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        self.commands.base_velocity.resampling_time_range = (5.0, 8.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5

        self.scene.leg_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


@configclass
class Atom01ParkourEnvCfg(Atom01ParkourRoughEnvCfg, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_shoe_config()


@configclass
class Atom01ParkourEnvCfg_PLAY(Atom01ParkourRoughEnvCfg_PLAY, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_shoe_config()
