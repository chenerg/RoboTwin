from copy import deepcopy

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


class place_blocks_color_pads(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.block_half_size = 0.025
        pad_half_size = (0.045, 0.045, 0.0005)

        pad_y = np.random.uniform(-0.18, -0.13)
        red_pad_x = np.random.choice([-0.12, 0.12])
        blue_pad_x = -red_pad_x
        self.red_pad = create_box(
            scene=self,
            pose=sapien.Pose(
                [red_pad_x, pad_y, 0.741],
                [1, 0, 0, 0],
            ),
            half_size=pad_half_size,
            color=(1, 0, 0),
            is_static=True,
            name="red_pad",
        )
        self.blue_pad = create_box(
            scene=self,
            pose=sapien.Pose(
                [blue_pad_x, pad_y, 0.741],
                [1, 0, 0, 0],
            ),
            half_size=pad_half_size,
            color=(0, 0, 1),
            is_static=True,
            name="blue_pad",
        )

        block_poses = []
        target_positions = [
            self.red_pad.get_pose().p[:2],
            self.blue_pad.get_pose().p[:2],
        ]

        for _ in range(2):
            for _ in range(100):
                candidate = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.04, 0.12],
                    zlim=[0.741 + self.block_half_size],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, np.pi / 3],
                )

                away_from_center = abs(candidate.p[0]) >= 0.06
                away_from_targets = all(
                    np.linalg.norm(candidate.p[:2] - target_position) >= 0.12
                    for target_position in target_positions
                )
                away_from_blocks = all(
                    np.linalg.norm(candidate.p[:2] - pose.p[:2]) >= 0.12
                    for pose in block_poses
                )

                if away_from_center and away_from_targets and away_from_blocks:
                    block_poses.append(deepcopy(candidate))
                    break
            else:
                raise RuntimeError("Failed to sample collision-free block poses.")

        self.red_block = create_box(
            scene=self,
            pose=block_poses[0],
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(1, 0, 0),
            name="red_block",
        )
        self.blue_block = create_box(
            scene=self,
            pose=block_poses[1],
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(0, 0, 1),
            name="blue_block",
        )

        self.add_prohibit_area(self.red_block, padding=0.06)
        self.add_prohibit_area(self.blue_block, padding=0.06)
        self.add_prohibit_area(self.red_pad, padding=0.05)
        self.add_prohibit_area(self.blue_pad, padding=0.05)

    def play_once(self):
        self.last_arm = None

        self.set_subtask(0)
        red_arm = self.pick_and_place_block(self.red_block, self.red_pad)

        self.set_subtask(1)
        blue_arm = self.pick_and_place_block(self.blue_block, self.blue_pad)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "red pad",
            "{C}": "blue block",
            "{D}": "blue pad",
            "{a}": str(red_arm),
            "{b}": str(blue_arm),
        }
        return self.info

    def pick_and_place_block(self, block, target_pad):
        arm_tag = ArmTag("left" if block.get_pose().p[0] < 0 else "right")
        grasp_action = self.grasp_actor(
            block,
            arm_tag=arm_tag,
            pre_grasp_dis=0.09,
        )

        if self.last_arm is not None and self.last_arm != arm_tag:
            self.move(
                grasp_action,
                self.back_to_origin(arm_tag.opposite),
            )
        else:
            self.move(grasp_action)

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))
        self.move(
            self.place_actor(
                block,
                arm_tag=arm_tag,
                target_pose=target_pad.get_functional_point(1),
                functional_point_id=0,
                pre_dis=0.07,
                dis=0.0,
                pre_dis_axis="fp",
                constrain="align",
            )
        )
        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=0.07,
                move_axis="arm",
            )
        )

        self.last_arm = arm_tag
        return arm_tag

    def check_success(self):
        return (
            self.block_on_pad(self.red_block, self.red_pad)
            and self.block_on_pad(self.blue_block, self.blue_pad)
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )

    def block_on_pad(self, block, pad):
        block_pose = block.get_pose().p
        pad_top_pose = pad.get_functional_point(1, "pose").p
        expected_block_center_z = pad_top_pose[2] + self.block_half_size

        return (
            np.linalg.norm(block_pose[:2] - pad_top_pose[:2]) < 0.03
            and abs(block_pose[2] - expected_block_center_z) < 0.015
        )
