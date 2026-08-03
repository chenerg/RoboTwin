"""Reusable expert policies for procedurally generated block tasks.

Concrete tasks remain small same-named classes in their own modules.  Keeping
the sampling, expert motion, and success checks here makes related tasks share
the same physical assumptions and prevents copy-pasted variants from drifting.
"""

from __future__ import annotations

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import ArmTag, create_box, rand_pose


COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 0.8, 0.0),
    "blue": (0.0, 0.2, 1.0),
    "yellow": (1.0, 0.85, 0.0),
    "orange": (1.0, 0.35, 0.0),
    "purple": (0.55, 0.15, 0.8),
    "gray": (0.45, 0.45, 0.45),
}


class PrimitiveBlockPolicy(Base_Task):
    """Common scene and motion helpers for the generated block policies."""

    block_half_size = 0.025
    pad_half_size = (0.05, 0.05, 0.0005)

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def _make_block(self, color, pose, name=None, is_static=False):
        return create_box(
            scene=self,
            pose=pose,
            half_size=(self.block_half_size,) * 3,
            color=COLORS[color],
            is_static=is_static,
            name=name or f"{color}_block",
        )

    def _make_pad(self, color, pose, name=None):
        return create_box(
            scene=self,
            pose=pose,
            half_size=self.pad_half_size,
            color=COLORS[color],
            is_static=True,
            name=name or f"{color}_pad",
        )

    def _sample_block_pose(self, occupied_xy=(), y_lim=(-0.04, 0.1)):
        """Sample a reachable pose at least 12 cm from occupied positions."""
        for _ in range(200):
            pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=list(y_lim),
                zlim=[0.741 + self.block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi / 3],
            )
            if abs(pose.p[0]) < 0.06:
                continue
            if any(np.linalg.norm(pose.p[:2] - np.asarray(xy)) < 0.12 for xy in occupied_xy):
                continue
            return pose
        raise RuntimeError("Failed to sample a collision-free block pose after 200 attempts.")

    def _surface_target(self, xy):
        """Return a FP0 target on the current (possibly randomized) table."""
        return [float(xy[0]), float(xy[1]), 0.741 + self.table_z_bias, 0, 1, 0, 0]

    @staticmethod
    def _arm_for(actor):
        return ArmTag("left" if actor.get_pose().p[0] < 0 else "right")

    def _pick_and_place(self, actor, target_pose, last_arm=None):
        arm_tag = self._arm_for(actor)
        grasp = self.grasp_actor(actor, arm_tag=arm_tag, pre_grasp_dis=0.09)
        if last_arm is not None and last_arm != arm_tag:
            self.move(grasp, self.back_to_origin(arm_tag.opposite))
        else:
            self.move(grasp)

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))
        self.move(
            self.place_actor(
                actor,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.07,
                dis=0.0,
                pre_dis_axis="fp",
                constrain="align",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))
        return arm_tag

    def _all_grippers_open(self):
        return self.is_left_gripper_open() and self.is_right_gripper_open()

    def _block_on_pad(self, block, pad):
        block_bottom = block.get_functional_point(0, "pose").p
        pad_top = pad.get_functional_point(1, "pose").p
        return np.linalg.norm(block_bottom[:2] - pad_top[:2]) < 0.032 and abs(block_bottom[2] - pad_top[2]) < 0.018


class PlaceBlockOnPadPolicy(PrimitiveBlockPolicy):
    """Pick one colored block and put it on a differently colored pad."""

    block_color = "red"
    pad_color = "blue"

    def load_actors(self):
        pad_x = float(np.random.choice([-0.12, 0.12]))
        pad_y = float(np.random.uniform(-0.19, -0.14))
        pad_pose = sapien.Pose([pad_x, pad_y, 0.741], [1, 0, 0, 0])
        block_pose = self._sample_block_pose([pad_pose.p[:2]])

        self.pad = self._make_pad(self.pad_color, pad_pose)
        self.block = self._make_block(self.block_color, block_pose)
        self.add_prohibit_area(self.pad, padding=0.05)
        self.add_prohibit_area(self.block, padding=0.06)

    def play_once(self):
        self.set_subtask(0)
        arm_tag = self._pick_and_place(self.block, self.pad.get_functional_point(1))
        self.info["info"] = {
            "{A}": f"{self.block_color} block",
            "{B}": f"{self.pad_color} pad",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        return self._block_on_pad(self.block, self.pad) and self._all_grippers_open()


class RelativeBlockPlacementPolicy(PrimitiveBlockPolicy):
    """Place one block in a configured planar relation to a reference block."""

    moving_color = "red"
    reference_color = "blue"
    relation = "left of"
    relation_offsets = {
        "left of": np.array([-0.09, 0.0]),
        "right of": np.array([0.09, 0.0]),
        "in front of": np.array([0.0, -0.09]),
        "behind": np.array([0.0, 0.09]),
    }

    def load_actors(self):
        offset = self.relation_offsets[self.relation]
        for _ in range(200):
            reference_xy = np.array([
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.13, -0.08),
            ])
            goal_xy = reference_xy + offset
            if abs(goal_xy[0]) <= 0.2 and -0.23 <= goal_xy[1] <= 0.04:
                break
        else:
            raise RuntimeError("Failed to sample a reachable relative-placement goal.")

        reference_pose = sapien.Pose(
            [reference_xy[0], reference_xy[1], 0.741 + self.block_half_size],
            [1, 0, 0, 0],
        )
        moving_pose = self._sample_block_pose([reference_xy, goal_xy], y_lim=(-0.01, 0.12))
        self.reference_block = self._make_block(
            self.reference_color,
            reference_pose,
            name=f"reference_{self.reference_color}_block",
            is_static=True,
        )
        self.moving_block = self._make_block(self.moving_color, moving_pose)
        self.goal_xy = goal_xy

        self.add_prohibit_area(self.reference_block, padding=0.06)
        self.add_prohibit_area(self.moving_block, padding=0.06)
        self.prohibited_area.append([
            goal_xy[0] - 0.05,
            goal_xy[1] - 0.05,
            goal_xy[0] + 0.05,
            goal_xy[1] + 0.05,
        ])

    def play_once(self):
        self.set_subtask(0)
        arm_tag = self._pick_and_place(self.moving_block, self._surface_target(self.goal_xy))
        self.info["info"] = {
            "{A}": f"{self.moving_color} block",
            "{B}": f"{self.reference_color} block",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        moving = self.moving_block.get_pose().p
        reference = self.reference_block.get_pose().p
        actual_offset = moving[:2] - reference[:2]
        desired_offset = self.relation_offsets[self.relation]
        axis = int(np.argmax(np.abs(desired_offset)))
        perpendicular_axis = 1 - axis
        signed_distance = actual_offset[axis] * np.sign(desired_offset[axis])
        relation_ok = 0.06 < signed_distance < 0.13 and abs(actual_offset[perpendicular_axis]) < 0.035
        on_table = abs(moving[2] - (0.741 + self.table_z_bias + self.block_half_size)) < 0.018
        return relation_ok and on_table and self._all_grippers_open()


class StackBlocksPolicy(PrimitiveBlockPolicy):
    """Move a bottom block to a base pad, then stack the top block on it."""

    top_color = "red"
    bottom_color = "blue"

    def load_actors(self):
        base_xy = np.array([0.0, np.random.uniform(-0.18, -0.14)])
        base_pose = sapien.Pose([base_xy[0], base_xy[1], 0.741], [1, 0, 0, 0])
        bottom_pose = self._sample_block_pose([base_xy])
        top_pose = self._sample_block_pose([base_xy, bottom_pose.p[:2]])

        self.base_pad = self._make_pad("gray", base_pose, name="stack_base")
        self.bottom_block = self._make_block(self.bottom_color, bottom_pose)
        self.top_block = self._make_block(self.top_color, top_pose)
        self.add_prohibit_area(self.base_pad, padding=0.05)
        self.add_prohibit_area(self.bottom_block, padding=0.06)
        self.add_prohibit_area(self.top_block, padding=0.06)

    def play_once(self):
        self.set_subtask(0)
        bottom_arm = self._pick_and_place(self.bottom_block, self.base_pad.get_functional_point(1))
        self.set_subtask(1)
        top_arm = self._pick_and_place(
            self.top_block,
            self.bottom_block.get_functional_point(1),
            last_arm=bottom_arm,
        )
        self.info["info"] = {
            "{A}": f"{self.top_color} block",
            "{B}": f"{self.bottom_color} block",
            "{a}": str(top_arm),
            "{b}": str(bottom_arm),
        }
        return self.info

    def check_success(self):
        top_bottom = self.top_block.get_functional_point(0, "pose").p
        bottom_top = self.bottom_block.get_functional_point(1, "pose").p
        return (
            self._block_on_pad(self.bottom_block, self.base_pad)
            and np.linalg.norm(top_bottom[:2] - bottom_top[:2]) < 0.028
            and abs(top_bottom[2] - bottom_top[2]) < 0.016
            and self._all_grippers_open()
        )


class RankBlocksPolicy(PrimitiveBlockPolicy):
    """Arrange three colored blocks in a configured left-to-right order."""

    color_order = ("red", "green", "blue")

    @staticmethod
    def _poses_are_ranked(poses):
        ordered = poses[0][0] < poses[1][0] < poses[2][0]
        aligned = max(pose[1] for pose in poses) - min(pose[1] for pose in poses) < 0.035
        return ordered and aligned

    def load_actors(self):
        for _ in range(200):
            block_poses = []
            occupied = []
            for _color in self.color_order:
                pose = self._sample_block_pose(occupied)
                block_poses.append(pose)
                occupied.append(pose.p[:2])
            if not self._poses_are_ranked([pose.p for pose in block_poses]):
                break
        else:
            raise RuntimeError("Failed to sample a ranking task that is not initially solved.")

        self.blocks = [
            self._make_block(color, pose)
            for color, pose in zip(self.color_order, block_poses)
        ]
        for block in self.blocks:
            self.add_prohibit_area(block, padding=0.05)

        target_y = float(np.random.uniform(-0.19, -0.14))
        self.target_poses = [
            self._surface_target((target_x, target_y))
            for target_x in (-0.09, 0.0, 0.09)
        ]
        self.prohibited_area.append([-0.15, target_y - 0.05, 0.15, target_y + 0.05])

    def play_once(self):
        arms = []
        last_arm = None
        for index, (block, target_pose) in enumerate(zip(self.blocks, self.target_poses)):
            self.set_subtask(index)
            last_arm = self._pick_and_place(block, target_pose, last_arm=last_arm)
            arms.append(last_arm)

        self.info["info"] = {
            "{A}": f"{self.color_order[0]} block",
            "{B}": f"{self.color_order[1]} block",
            "{C}": f"{self.color_order[2]} block",
            "{a}": str(arms[0]),
            "{b}": str(arms[1]),
            "{c}": str(arms[2]),
        }
        return self.info

    def check_success(self):
        positions = [block.get_pose().p for block in self.blocks]
        adjacent_gaps = [positions[i + 1][0] - positions[i][0] for i in range(2)]
        on_table = all(
            abs(position[2] - (0.741 + self.table_z_bias + self.block_half_size)) < 0.018
            for position in positions
        )
        return (
            self._poses_are_ranked(positions)
            and all(0.05 < gap < 0.14 for gap in adjacent_gaps)
            and on_table
            and self._all_grippers_open()
        )


class PlaceBlocksOnPadsPolicy(PrimitiveBlockPolicy):
    """Place two blocks on configured matching or crossed colored pads."""

    block_colors = ("red", "blue")
    pad_colors = ("red", "blue")
    target_indices = (0, 1)

    def load_actors(self):
        pad_y = float(np.random.uniform(-0.19, -0.14))
        pad_xs = (-0.12, 0.12)
        self.pads = [
            self._make_pad(color, sapien.Pose([x, pad_y, 0.741], [1, 0, 0, 0]))
            for color, x in zip(self.pad_colors, pad_xs)
        ]
        occupied = [pad.get_pose().p[:2] for pad in self.pads]
        block_poses = []
        for _color in self.block_colors:
            pose = self._sample_block_pose(occupied)
            block_poses.append(pose)
            occupied.append(pose.p[:2])
        self.blocks = [
            self._make_block(color, pose)
            for color, pose in zip(self.block_colors, block_poses)
        ]
        for actor in [*self.pads, *self.blocks]:
            self.add_prohibit_area(actor, padding=0.055)

    def play_once(self):
        arms = []
        last_arm = None
        for index, (block, target_index) in enumerate(zip(self.blocks, self.target_indices)):
            self.set_subtask(index)
            target = self.pads[target_index].get_functional_point(1)
            last_arm = self._pick_and_place(block, target, last_arm=last_arm)
            arms.append(last_arm)

        self.info["info"] = {
            "{A}": f"{self.block_colors[0]} block",
            "{B}": f"{self.pad_colors[self.target_indices[0]]} pad",
            "{C}": f"{self.block_colors[1]} block",
            "{D}": f"{self.pad_colors[self.target_indices[1]]} pad",
            "{a}": str(arms[0]),
            "{b}": str(arms[1]),
        }
        return self.info

    def check_success(self):
        return (
            all(
                self._block_on_pad(block, self.pads[target_index])
                for block, target_index in zip(self.blocks, self.target_indices)
            )
            and self._all_grippers_open()
        )
