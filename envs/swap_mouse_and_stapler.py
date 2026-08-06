"""Swap a mouse and a stapler across the table.

Self-contained expert policy (no dependency on _household_task_policy).
Direct 3-move choreography: stage the mouse, move the stapler straight into
the mouse's original spot, then drop the mouse into the stapler's spot.
Both objects are placed flat (z-up target quaternion) with free yaw.
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


class swap_mouse_and_stapler(Base_Task):
    object_a_spec = ("047_mouse", [0, 1, 2])
    object_b_spec = ("048_stapler", [0, 1, 2, 3, 4, 5, 6])

    # Match place_mouse_pad for both mouse handoffs.  The mouse has no
    # functional point, so both placements still align its actor root.
    object_a_grasp_pre_dis = 0.1
    object_a_lift_height = 0.1
    object_a_place_pre_dis = 0.07
    object_a_place_dis = 0.005
    object_a_place_constrain = "free"
    object_a_target_quaternion = (0, 0, 0, 1)

    # Central staging slot: reachable by both arms and well inside the head
    # camera view (the default right-edge slot sits at the FOV boundary).
    staging_xy = (0.0, -0.14)

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def _create_object(self, spec, pose):
        model_name = spec[0]
        allowed_ids = spec[1] if len(spec) > 1 and spec[1] is not None else _model_ids(model_name)
        model_id = int(np.random.choice(allowed_ids))
        actor = create_actor(
            scene=self,
            pose=pose,
            modelname=model_name,
            model_id=model_id,
            convex=True,
        )
        actor.set_mass(0.04)
        return actor, model_name, model_id

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
            [b_pose.p[0], b_pose.p[1], a_pose.p[2]], self.object_a_target_quaternion
        )
        # b_pose.q = (0.5,0.5,0.5,0.5) has its z-axis pointing at world +x,
        # which would tilt the stapler onto its side.  Use a z-up quaternion
        # so the flush (dis=0) placement keeps it flat.
        self.object_b_goal = sapien.Pose(
            [a_pose.p[0], a_pose.p[1], b_pose.p[2]], (0, 0, 0, 1)
        )

        self.add_prohibit_area(self.object_a, padding=0.07)
        self.add_prohibit_area(self.object_b, padding=0.07)
        staging_padding = 0.055
        self.prohibited_area.append(
            [
                self.staging_xy[0] - staging_padding,
                self.staging_xy[1] - staging_padding,
                self.staging_xy[0] + staging_padding,
                self.staging_xy[1] + staging_padding,
            ]
        )

    def _pick_place_root(
        self,
        actor,
        arm_tag,
        target_pose,
        stage_prefix,
        *,
        grasp_pre_dis=0.1,
        lift_height=0.11,
        place_pre_dis=0.08,
        place_dis=0.005,
        place_constrain="free",
        retreat_height=0.07,
        return_to_origin=False,
    ):
        self.run_action_stage(
            f"{stage_prefix}_grasp",
            lambda: self.grasp_actor(
                actor,
                arm_tag=arm_tag,
                pre_grasp_dis=grasp_pre_dis,
            ),
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
                constrain=place_constrain,
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

    def _relocate_object_b(self, arm_tag, target_pose, stage_prefix, *, return_to_origin=False):
        # Move the stapler exactly like move_stapler_pad: a direct grasp and
        # a lift along the gripper's local approach axis, instead of the
        # staged cycle used by the shared swap policy.  Placement keeps the
        # target quaternion for flatness but leaves yaw free like the mouse.
        self.move(self.grasp_actor(self.object_b, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))
        self.move(
            self.place_actor(
                self.object_b,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=0.1,
                dis=0.0,
                constrain="free",
            ))
        self.move(self.move_by_displacement(arm_tag, z=0.08))
        if return_to_origin:
            self.move(self.back_to_origin(arm_tag))

    def play_once(self):
        # Direct 3-move swap, mouse first: stage the mouse out of its slot,
        # move the stapler straight into the mouse's original spot, then drop
        # the mouse into the stapler's original spot.
        a_pose = self.object_a.get_pose()
        staging_pose_a = sapien.Pose(
            [self.staging_xy[0], self.staging_xy[1], a_pose.p[2]],
            self.object_a_target_quaternion,
        )

        self.set_subtask(0)
        self._pick_place_root(
            self.object_a,
            self.left_arm,
            staging_pose_a,
            "move_mouse_to_staging",
            grasp_pre_dis=self.object_a_grasp_pre_dis,
            lift_height=self.object_a_lift_height,
            place_pre_dis=self.object_a_place_pre_dis,
            place_dis=self.object_a_place_dis,
            place_constrain=self.object_a_place_constrain,
            retreat_height=0.08,
            return_to_origin=True,
        )

        self.set_subtask(1)
        self._relocate_object_b(
            self.right_arm,
            self.object_b_goal,
            "move_stapler_to_mouse_spot",
            return_to_origin=True,
        )

        self.set_subtask(2)
        self._pick_place_root(
            self.object_a,
            self.left_arm,
            self.object_a_goal,
            "move_mouse_to_stapler_spot",
            grasp_pre_dis=self.object_a_grasp_pre_dis,
            lift_height=self.object_a_lift_height,
            place_pre_dis=self.object_a_place_pre_dis,
            place_dis=self.object_a_place_dis,
            place_constrain=self.object_a_place_constrain,
            retreat_height=0.08,
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
