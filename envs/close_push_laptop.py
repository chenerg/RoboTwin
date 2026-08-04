import numpy as np
import transforms3d as t3d

from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .close_laptop import close_laptop
from .utils import *


class close_push_laptop(close_laptop):

    CLOSED_ANGLE_TOLERANCE_RAD = np.deg2rad(5.0)
    CLOSING_JOINT_FRACTIONS = (0.40, 0.30, 0.20, 0.10, 0.05, 0.02, 0.0)
    PUSH_GRIPPER_POS = 0.3
    GRIPPER_REACH = 0.12
    PUSH_PENETRATION = 0.004

    def play_once(self):
        face_prod = get_face_prod(self.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        self.arm_tag = ArmTag("left" if face_prod > 0 else "right")
        self.set_subtask(0)
        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{a}": str(self.arm_tag),
        }
        self.info["laptop_model_id"] = int(self.model_id)
        self.info["laptop_joint_trace"] = []

        # Shape the gripper into a compact pusher before touching the screen.
        # CP0 is on the upper screen; CP1 is too close to the hinge and table.
        self.run_action_stage(
            "shape_gripper_for_top_down_push",
            lambda: self.close_gripper(
                self.arm_tag,
                pos=self.PUSH_GRIPPER_POS,
            ),
        )
        if not self.plan_success:
            return self.info

        # Approach CP0 from above. Recompute the contact point at every stage
        # because the passive screen can move as the gripper gets close.
        approach_start = self._joint_fraction()
        for stage_name, clearance in (
            ("approach_screen_from_above", 0.10),
            ("approach_screen_from_above_near", 0.03),
            ("contact_screen_from_above", 0.0),
        ):
            before = self._joint_fraction()
            self.run_action_stage(
                stage_name,
                lambda clearance=clearance: self._move_to_dynamic_top_down_push(
                    clearance,
                ),
            )
            if not self.plan_success:
                return self.info
            after = self._joint_fraction()
            self._record_joint_motion(stage_name, before, after)
            if after > approach_start + self.APPROACH_OPENING_TOLERANCE:
                self._fail_joint_motion(
                    stage_name,
                    before,
                    after,
                    "screen was pushed open during top-down approach",
                )
                return self.info

        # Push the upper screen through positions sampled from decreasing hinge
        # angles. The hinge stays passive; closure comes from physical contact.
        push_start_fraction = self._joint_fraction()
        previous_fraction = push_start_fraction
        for step, (target_fraction, target_contact_matrix) in enumerate(
            self._closing_contact_path,
            start=1,
        ):
            if self._is_closed_pose():
                self.stage_success_tag = True
                break
            if target_fraction >= previous_fraction - self.MIN_CLOSING_PROGRESS:
                continue

            stage_name = f"close_lid_step_{step}"
            self.run_action_stage(
                stage_name,
                lambda target_contact_matrix=target_contact_matrix: (
                    self._move_to_cached_top_down_push(target_contact_matrix)
                ),
            )
            if not self.plan_success:
                break

            current_fraction = self._joint_fraction()
            self._record_joint_motion(stage_name, previous_fraction, current_fraction)
            if current_fraction > previous_fraction + self.STEP_OPENING_TOLERANCE:
                self._fail_joint_motion(
                    stage_name,
                    previous_fraction,
                    current_fraction,
                    "laptop joint moved in the opening direction",
                )
                break
            if current_fraction > push_start_fraction + self.TOTAL_OPENING_TOLERANCE:
                self._fail_joint_motion(
                    stage_name,
                    previous_fraction,
                    current_fraction,
                    "laptop joint accumulated motion in the opening direction",
                )
                break
            if current_fraction >= previous_fraction - self.MIN_CLOSING_PROGRESS:
                self._fail_joint_motion(
                    stage_name,
                    previous_fraction,
                    current_fraction,
                    "laptop joint did not make closing progress",
                )
                break
            previous_fraction = current_fraction

        if self.plan_success and self._is_closed_pose():
            self.stage_success_tag = True

        if self.plan_success:
            self.run_action_stage(
                "release_screen",
                lambda: self.open_gripper(self.arm_tag),
            )
            self.run_action_stage(
                "retreat_above_screen",
                lambda: self.move_by_displacement(self.arm_tag, z=0.08),
            )
            self.run_action_stage(
                "return_arm_to_origin",
                lambda: self.back_to_origin(self.arm_tag),
            )
        return self.info

    def _top_down_push_pose(self, contact_matrix, clearance):
        push_quat = np.asarray(GRASP_DIRECTION_DIC["top_down"], dtype=np.float64)
        push_rotation = t3d.quaternions.quat2mat(push_quat)
        push_position = np.array(
            contact_matrix[:3, 3],
            dtype=np.float64,
            copy=True,
        )
        push_position += push_rotation @ np.array(
            [-self.GRIPPER_REACH - clearance, 0.0, 0.0],
            dtype=np.float64,
        )
        return push_position.tolist() + push_quat.tolist()

    def _move_to_dynamic_top_down_push(self, clearance):
        if not self.plan_success:
            return self.arm_tag, []
        if not self.need_plan:
            return self.move_to_pose(self.arm_tag, [0.0] * 7)

        contact_matrix = self.laptop.get_contact_point(
            self.SCREEN_CONTACT_POINT_ID,
            "matrix",
        )
        if contact_matrix is None:
            self.plan_success = False
            return self.arm_tag, []
        return self.move_to_pose(
            self.arm_tag,
            self._top_down_push_pose(contact_matrix, clearance),
        )

    def _move_to_cached_top_down_push(self, target_contact_matrix):
        if not self.plan_success:
            return self.arm_tag, []
        if not self.need_plan:
            return self.move_to_pose(self.arm_tag, [0.0] * 7)

        return self.move_to_pose(
            self.arm_tag,
            self._top_down_push_pose(
                target_contact_matrix,
                -self.PUSH_PENETRATION,
            ),
        )

    def _is_closed_pose(self):
        closed_qpos = float(self.laptop.get_qlimits()[0][0])
        current_qpos = float(self.laptop.get_qpos()[0])
        return abs(current_qpos - closed_qpos) < self.CLOSED_ANGLE_TOLERANCE_RAD
