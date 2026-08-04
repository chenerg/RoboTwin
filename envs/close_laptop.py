import numpy as np
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import *


class close_laptop(Base_Task):

    INITIAL_JOINT_FRACTION = 0.5
    MAX_CLOSED_JOINT_FRACTION = 0.3

    # Keep the hinge passive.  These values only model a screen with realistic
    # inertia and hinge resistance; no drive target or positional stiffness is
    # applied during the manipulation.
    LAPTOP_LINK_MASS = 0.2
    HINGE_DAMPING = 2.0
    HINGE_FRICTION = 0.2
    SCREEN_CONTACT_POINT_ID = 0

    APPROACH_OPENING_TOLERANCE = 0.05
    STEP_OPENING_TOLERANCE = 0.01
    TOTAL_OPENING_TOLERANCE = 0.02
    MIN_CLOSING_PROGRESS = 0.003
    CLOSING_JOINT_FRACTIONS = (0.45, 0.40, 0.35, 0.30, 0.25)

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.model_name = "015_laptop"
        self.model_id = np.random.randint(0, 11)
        self.laptop = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[-0.1, 0.05],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 3],
            qpos=[0.7, 0, 0, 0.7],
            fix_root_link=True,
        )
        limit = self.laptop.get_qlimits()[0]
        self.laptop.set_qpos(
            [limit[0] + (limit[1] - limit[0]) * self.INITIAL_JOINT_FRACTION]
        )
        self.laptop.set_mass(self.LAPTOP_LINK_MASS)
        self.laptop.set_properties(
            damping=self.HINGE_DAMPING,
            stiffness=0,
            friction=self.HINGE_FRICTION,
        )
        self._cache_closing_contact_path()
        self.add_prohibit_area(self.laptop, padding=0.1)

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

        # A generic grasp_actor computes both poses before moving.  A passive
        # laptop screen may move during the approach, so approach in short
        # stages and recompute the upper-screen CP0 from its current pose each
        # time.  CP1 is lower and too close to the hinge/table on these assets.
        approach_start = self._joint_fraction()
        for stage_name, pre_dis in (
            ("approach_screen_cp0", 0.12),
            ("approach_screen_cp0_near", 0.04),
            ("approach_screen_cp0_fine", 0.02),
            ("approach_screen_cp0_final", 0.0),
        ):
            before = self._joint_fraction()
            self.run_action_stage(
                stage_name,
                lambda pre_dis=pre_dis: self._move_to_dynamic_screen_grasp(pre_dis),
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
                    "screen was pushed open during grasp approach",
                )
                return self.info

        before = self._joint_fraction()
        self.run_action_stage(
            "close_gripper_on_screen",
            lambda: self.close_gripper(self.arm_tag),
        )
        if not self.plan_success:
            return self.info
        after = self._joint_fraction()
        self._record_joint_motion("close_gripper_on_screen", before, after)
        if after > approach_start + self.APPROACH_OPENING_TOLERANCE:
            self._fail_joint_motion(
                "close_gripper_on_screen",
                before,
                after,
                "screen was pushed open while closing the gripper",
            )
            return self.info

        grasp_fraction = after
        grasp_contact_matrix = self.laptop.get_contact_point(
            self.SCREEN_CONTACT_POINT_ID,
            "matrix",
        )
        grasp_ee_matrix = self._current_ee_matrix()

        # Move the held CP0 along poses sampled from decreasing joint angles.
        # The samples provide geometry only: the joint remains passive and the
        # robot must physically rotate the screen through contact.
        previous_fraction = grasp_fraction
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
                lambda target_contact_matrix=target_contact_matrix: self._move_along_hinge_arc(
                    grasp_contact_matrix,
                    grasp_ee_matrix,
                    target_contact_matrix,
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
            if current_fraction > grasp_fraction + self.TOTAL_OPENING_TOLERANCE:
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
                "return_arm_to_origin",
                lambda: self.back_to_origin(self.arm_tag),
            )
        return self.info

    def _cache_closing_contact_path(self):
        """Cache upper-screen contact geometry before robot contact."""
        original_qpos = np.array(self.laptop.get_qpos(), dtype=np.float64)
        limit = self.laptop.get_qlimits()[0]
        span = float(limit[1] - limit[0])
        self._closing_contact_path = []
        for fraction in self.CLOSING_JOINT_FRACTIONS:
            self.laptop.set_qpos([limit[0] + span * fraction])
            contact_matrix = np.array(
                self.laptop.get_contact_point(
                    self.SCREEN_CONTACT_POINT_ID,
                    "matrix",
                ),
                dtype=np.float64,
                copy=True,
            )
            self._closing_contact_path.append((fraction, contact_matrix))
        self.laptop.set_qpos(original_qpos)
        self.laptop.set_qvel(np.zeros_like(original_qpos))

    def _move_to_dynamic_screen_grasp(self, pre_dis):
        if not self.plan_success:
            return self.arm_tag, []
        if not self.need_plan:
            return self.move_to_pose(self.arm_tag, [0.0] * 7)

        target_pose = self.get_grasp_pose(
            self.laptop,
            arm_tag=self.arm_tag,
            contact_point_id=self.SCREEN_CONTACT_POINT_ID,
            pre_dis=pre_dis,
        )
        if target_pose is None:
            self.plan_success = False
            return self.arm_tag, []
        return self.move_to_pose(self.arm_tag, target_pose)

    def _move_along_hinge_arc(
        self,
        grasp_contact_matrix,
        grasp_ee_matrix,
        target_contact_matrix,
    ):
        if not self.plan_success:
            return self.arm_tag, []
        if not self.need_plan:
            return self.move_to_pose(self.arm_tag, [0.0] * 7)

        contact_to_ee = np.linalg.inv(grasp_contact_matrix) @ grasp_ee_matrix
        target_ee_matrix = target_contact_matrix @ contact_to_ee
        target_pose = target_ee_matrix[:3, 3].tolist()
        target_pose += t3d.quaternions.mat2quat(target_ee_matrix[:3, :3]).tolist()
        return self.move_to_pose(self.arm_tag, target_pose)

    def _current_ee_matrix(self):
        pose = np.asarray(self.get_arm_pose(self.arm_tag), dtype=np.float64)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = t3d.quaternions.quat2mat(pose[3:])
        matrix[:3, 3] = pose[:3]
        return matrix

    def _record_joint_motion(self, stage_name, before, after):
        self.info["laptop_joint_trace"].append(
            {
                "stage": stage_name,
                "before": float(before),
                "after": float(after),
                "delta": float(after - before),
            }
        )

    def _fail_joint_motion(self, stage_name, before, after, reason):
        self.plan_success = False
        self.report_action_stage_failure(
            stage_name,
            (
                f"{reason}; model_id={self.model_id} "
                f"q_before={before:.4f} q_after={after:.4f} "
                f"delta={after - before:+.4f}"
            ),
        )

    def _joint_fraction(self):
        limit = self.laptop.get_qlimits()[0]
        span = max(float(limit[1] - limit[0]), 1e-6)
        return float((self.laptop.get_qpos()[0] - limit[0]) / span)

    def _is_closed_pose(self):
        return self._joint_fraction() <= self.MAX_CLOSED_JOINT_FRACTION

    def check_success(self):
        if not hasattr(self, "arm_tag"):
            return False
        return (
            self.stage_success_tag
            and self._is_closed_pose()
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
