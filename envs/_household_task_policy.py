"""Shared expert policies for household-object rearrangement tasks.

The concrete task modules intentionally contain only declarative asset choices.
This keeps sampling, action staging, success checks, and language metadata
consistent across the fifteen tasks built from assets already exercised by the
original RoboTwin task suite.
"""

from pathlib import Path

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


DEFAULT_OBJECT_QUATERNION = (0.5, 0.5, 0.5, 0.5)


def _pose(x, y, quaternion=DEFAULT_OBJECT_QUATERNION):
    return rand_pose(
        xlim=[x, x],
        ylim=[y, y],
        qpos=quaternion,
        rotate_rand=False,
    )


def _model_ids(model_name):
    ids = []
    for path in (Path("assets/objects") / model_name).glob("model_data*.json"):
        suffix = path.stem[len("model_data"):]
        if suffix.isdigit():
            ids.append(int(suffix))
    if not ids:
        raise ValueError(f"No rigid models found for {model_name}")
    return sorted(ids)


def _asset_info(model_name, model_id):
    return f"{model_name}/base{model_id}"


class HouseholdPolicy(Base_Task):
    """Small utilities shared by the three household task families."""

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def _create_object(self, spec, pose, *, is_static=False, scale=(1, 1, 1)):
        model_name = spec[0]
        allowed_ids = spec[1] if len(spec) > 1 and spec[1] is not None else _model_ids(model_name)
        model_id = int(np.random.choice(allowed_ids))
        actor = create_actor(
            scene=self,
            pose=pose,
            modelname=model_name,
            model_id=model_id,
            scale=scale,
            convex=True,
            is_static=is_static,
        )
        if not is_static:
            actor.set_mass(0.04)
        return actor, model_name, model_id

    def _pick_place_root(
        self,
        actor,
        arm_tag,
        target_pose,
        stage_prefix,
        *,
        lift_height=0.11,
        place_pre_dis=0.08,
        place_dis=0.005,
        retreat_height=0.07,
        return_to_origin=False,
    ):
        self.run_action_stage(
            f"{stage_prefix}_grasp",
            lambda: self.grasp_actor(actor, arm_tag=arm_tag, pre_grasp_dis=0.1),
        )
        self.run_action_stage(
            f"{stage_prefix}_lift",
            lambda: self.move_by_displacement(arm_tag, z=lift_height),
        )
        self.run_action_stage(
            f"{stage_prefix}_place",
            lambda: self.place_actor(
                actor,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=place_pre_dis,
                dis=place_dis,
                constrain="free",
            ),
        )
        self.run_action_stage(
            f"{stage_prefix}_retreat",
            lambda: self.move_by_displacement(arm_tag, z=retreat_height),
        )
        if return_to_origin:
            self.run_action_stage(
                f"{stage_prefix}_return_to_origin",
                lambda: self.back_to_origin(arm_tag),
            )


class SwapHouseholdObjectsPolicy(HouseholdPolicy):
    """Swap objects across the table through a shared raised relay platform."""

    object_a_spec = None
    object_b_spec = None

    def load_actors(self):
        pose_a = _pose(-0.22, 0.03)
        pose_b = _pose(0.22, 0.03)

        self.object_a, self.object_a_name, self.object_a_id = self._create_object(
            self.object_a_spec, pose_a
        )
        self.object_b, self.object_b_name, self.object_b_id = self._create_object(
            self.object_b_spec, pose_b
        )
        self.object_a.set_name("swap_object_a")
        self.object_b.set_name("swap_object_b")
        self.left_arm = ArmTag("left")
        self.right_arm = ArmTag("right")

        a_pose = self.object_a.get_pose()
        b_pose = self.object_b.get_pose()
        self.object_a_goal = sapien.Pose(
            [b_pose.p[0], b_pose.p[1], a_pose.p[2]], a_pose.q
        )
        self.object_b_goal = sapien.Pose(
            [a_pose.p[0], a_pose.p[1], b_pose.p[2]], b_pose.q
        )
        # Move B out of its destination before relaying A.  The slot remains
        # completely inside the right arm workspace.
        self.staging_pose = sapien.Pose(
            [0.285, -0.14, b_pose.p[2]], b_pose.q
        )

        # Both arms use this raised front-center platform sequentially.  It
        # avoids cross-body placements and also supports objects such as the
        # bell that expose only one grasp point, so a direct handover would be
        # under-constrained or gripper-to-gripper colliding.
        platform_center_z = 0.781 + self.table_z_bias
        self.transfer_platform = create_box(
            scene=self.scene,
            pose=sapien.Pose([0.0, -0.16, platform_center_z], [1, 0, 0, 0]),
            half_size=(0.085, 0.065, 0.04),
            color=(0.55, 0.55, 0.55),
            is_static=True,
            name="swap_transfer_platform",
        )
        platform_height = 0.08
        self.object_a_transfer_pose = sapien.Pose(
            [0.0, -0.16, a_pose.p[2] + platform_height], a_pose.q
        )
        self.object_b_transfer_pose = sapien.Pose(
            [0.0, -0.16, b_pose.p[2] + platform_height], b_pose.q
        )

        self.add_prohibit_area(self.object_a, padding=0.07)
        self.add_prohibit_area(self.object_b, padding=0.07)
        self.add_prohibit_area(self.transfer_platform, padding=0.04)
        staging_padding = 0.055
        self.prohibited_area.append(
            [
                self.staging_pose.p[0] - staging_padding,
                self.staging_pose.p[1] - staging_padding,
                self.staging_pose.p[0] + staging_padding,
                self.staging_pose.p[1] + staging_padding,
            ]
        )

    def play_once(self):
        # Free the right-side destination before moving A across the table.
        self.set_subtask(0)
        self._pick_place_root(
            self.object_b,
            self.right_arm,
            self.staging_pose,
            "move_right_object_to_staging",
            lift_height=0.16,
            place_pre_dis=0.12,
            place_dis=0.02,
            retreat_height=0.1,
            return_to_origin=True,
        )

        # Relay A from the left arm to the right arm through the platform.
        self.set_subtask(1)
        self._pick_place_root(
            self.object_a,
            self.left_arm,
            self.object_a_transfer_pose,
            "relay_left_object_to_center",
            lift_height=0.18,
            place_pre_dis=0.12,
            place_dis=0.02,
            retreat_height=0.1,
            return_to_origin=True,
        )
        self.set_subtask(2)
        self._pick_place_root(
            self.object_a,
            self.right_arm,
            self.object_a_goal,
            "move_left_object_to_right_spot",
            lift_height=0.12,
            place_pre_dis=0.12,
            place_dis=0.02,
            retreat_height=0.1,
            return_to_origin=True,
        )

        # Relay B in the opposite direction after A reaches the right side.
        self.set_subtask(3)
        self._pick_place_root(
            self.object_b,
            self.right_arm,
            self.object_b_transfer_pose,
            "relay_right_object_to_center",
            lift_height=0.18,
            place_pre_dis=0.12,
            place_dis=0.02,
            retreat_height=0.1,
            return_to_origin=True,
        )
        self.set_subtask(4)
        self._pick_place_root(
            self.object_b,
            self.left_arm,
            self.object_b_goal,
            "move_right_object_to_left_spot",
            lift_height=0.12,
            place_pre_dis=0.12,
            place_dis=0.02,
            retreat_height=0.1,
        )
        self.info["info"] = {
            "{A}": _asset_info(self.object_a_name, self.object_a_id),
            "{B}": _asset_info(self.object_b_name, self.object_b_id),
            "{a}": str(self.left_arm),
            "{b}": str(self.right_arm),
        }
        return self.info

    def check_success(self):
        if not hasattr(self, "object_a_goal"):
            return False
        a_distance = np.linalg.norm(
            self.object_a.get_pose().p[:2] - self.object_a_goal.p[:2]
        )
        b_distance = np.linalg.norm(
            self.object_b.get_pose().p[:2] - self.object_b_goal.p[:2]
        )
        return (
            a_distance < 0.04
            and b_distance < 0.04
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )


class ArrangeHouseholdObjectsPolicy(HouseholdPolicy):
    """Move three heterogeneous objects onto an ordered front-to-back row."""

    object_specs = ()
    pad_colors = ((0.9, 0.2, 0.2), (0.2, 0.75, 0.25), (0.2, 0.35, 0.9))

    def load_actors(self):
        if len(self.object_specs) != 3:
            raise ValueError("ArrangeHouseholdObjectsPolicy requires three object specs")

        side = int(np.random.choice([-1, 1]))
        self.arm_tag = ArmTag("left" if side < 0 else "right")
        source_y = np.random.permutation([-0.17, -0.03, 0.11])
        target_y = [-0.18, -0.04, 0.10]
        self.objects = []
        self.object_metadata = []
        self.goal_poses = []
        self.target_pads = []

        for index, spec in enumerate(self.object_specs):
            actor, name, model_id = self._create_object(
                spec, _pose(side * 0.245, float(source_y[index]))
            )
            actor.set_name(f"ordered_object_{index}")
            goal_pose = sapien.Pose(
                [side * 0.09, target_y[index], actor.get_pose().p[2]],
                actor.get_pose().q,
            )
            pad = create_box(
                scene=self.scene,
                pose=sapien.Pose(
                    [goal_pose.p[0], goal_pose.p[1], 0.741 + self.table_z_bias],
                    [1, 0, 0, 0],
                ),
                half_size=(0.045, 0.04, 0.0005),
                color=self.pad_colors[index],
                is_static=True,
                name=f"order_pad_{index}",
            )
            self.objects.append(actor)
            self.object_metadata.append((name, model_id))
            self.goal_poses.append(goal_pose)
            self.target_pads.append(pad)
            self.add_prohibit_area(actor, padding=0.06)
            self.add_prohibit_area(pad, padding=0.045)

    def play_once(self):
        for index, (actor, goal_pose) in enumerate(
            zip(self.objects, self.goal_poses)
        ):
            self.set_subtask(index)
            self._pick_place_root(
                actor,
                self.arm_tag,
                goal_pose,
                f"place_ordered_object_{index + 1}",
            )

        self.info["info"] = {
            "{A}": _asset_info(*self.object_metadata[0]),
            "{B}": _asset_info(*self.object_metadata[1]),
            "{C}": _asset_info(*self.object_metadata[2]),
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        if not hasattr(self, "goal_poses"):
            return False
        placed = all(
            np.linalg.norm(actor.get_pose().p[:2] - goal.p[:2]) < 0.04
            for actor, goal in zip(self.objects, self.goal_poses)
        )
        return (
            placed
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )


class PlaceHouseholdPairPolicy(HouseholdPolicy):
    """Place two different household objects on two designated target points."""

    source_specs = ()
    source_functional_points = (None, None)
    target_spec = None
    target_ids = (0, 1)
    separate_targets = False
    target_quaternion = DEFAULT_OBJECT_QUATERNION
    target_scale = None
    target_threshold = 0.055
    lift_height = 0.16

    def load_actors(self):
        if len(self.source_specs) != 2:
            raise ValueError("PlaceHouseholdPairPolicy requires two source specs")

        source_poses = (_pose(-0.245, 0.025), _pose(0.245, 0.025))
        self.sources = []
        self.source_metadata = []
        for index, (spec, pose) in enumerate(zip(self.source_specs, source_poses)):
            actor, name, model_id = self._create_object(spec, pose)
            actor.set_name(f"pair_source_{index}")
            self.sources.append(actor)
            self.source_metadata.append((name, model_id))
            self.add_prohibit_area(actor, padding=0.065)

        self.targets = []
        self.target_metadata = []
        if self.separate_targets:
            for index, x in enumerate((-0.11, 0.11)):
                target, name, model_id = self._create_object(
                    self.target_spec,
                    _pose(x, -0.155, self.target_quaternion),
                    is_static=True,
                )
                target.set_name(f"pair_target_{index}")
                self.targets.append(target)
                self.target_metadata.append((name, model_id))
                self.add_prohibit_area(target, padding=0.055)
        else:
            target, name, model_id = self._create_object(
                self.target_spec,
                _pose(0.0, -0.145, self.target_quaternion),
                is_static=True,
                scale=self.target_scale or (1, 1, 1),
            )
            target.set_name("pair_shared_target")
            self.targets = [target, target]
            self.target_metadata = [(name, model_id), (name, model_id)]
            self.add_prohibit_area(target, padding=0.09)

        self.target_poses = [
            self.targets[index].get_functional_point(self.target_ids[index], "pose")
            for index in range(2)
        ]

    def play_once(self):
        left_arm = ArmTag("left")
        right_arm = ArmTag("right")
        self.set_subtask(0)
        self.run_action_stage(
            "grasp_household_pair",
            lambda: self.grasp_actor(
                self.sources[0], arm_tag=left_arm, pre_grasp_dis=0.1
            ),
            lambda: self.grasp_actor(
                self.sources[1], arm_tag=right_arm, pre_grasp_dis=0.1
            ),
        )
        self.run_action_stage(
            "lift_household_pair",
            lambda: self.move_by_displacement(left_arm, z=self.lift_height),
            lambda: self.move_by_displacement(right_arm, z=self.lift_height),
        )

        self.set_subtask(1)
        self.run_action_stage(
            "place_left_household_object",
            lambda: self.place_actor(
                self.sources[0],
                arm_tag=left_arm,
                target_pose=self.target_poses[0],
                functional_point_id=self.source_functional_points[0],
                pre_dis=0.09,
                dis=0.005,
                constrain="free",
            ),
        )
        self.run_action_stage(
            "retreat_left_arm",
            lambda: self.move_by_displacement(left_arm, z=0.08),
        )

        self.set_subtask(2)
        self.run_action_stage(
            "place_right_household_object",
            lambda: self.back_to_origin(left_arm),
            lambda: self.place_actor(
                self.sources[1],
                arm_tag=right_arm,
                target_pose=self.target_poses[1],
                functional_point_id=self.source_functional_points[1],
                pre_dis=0.09,
                dis=0.005,
                constrain="free",
            ),
        )
        self.run_action_stage(
            "retreat_right_arm",
            lambda: self.move_by_displacement(right_arm, z=0.08),
        )

        episode_info = {
            "{A}": _asset_info(*self.source_metadata[0]),
            "{B}": _asset_info(*self.source_metadata[1]),
            "{C}": _asset_info(*self.target_metadata[0]),
        }
        if self.separate_targets:
            episode_info["{D}"] = _asset_info(*self.target_metadata[1])
        self.info["info"] = episode_info
        return self.info

    def _source_anchor(self, index):
        functional_point = self.source_functional_points[index]
        if functional_point is None:
            return self.sources[index].get_pose().p
        return self.sources[index].get_functional_point(functional_point, "pose").p

    def check_success(self):
        if not hasattr(self, "target_poses"):
            return False
        placed = all(
            np.linalg.norm(
                self._source_anchor(index)[:2] - self.target_poses[index].p[:2]
            )
            < self.target_threshold
            and self._source_anchor(index)[2] > self.target_poses[index].p[2] - 0.02
            for index in range(2)
        )
        return (
            placed
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
