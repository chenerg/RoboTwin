import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


class insert_markpen_into_pencup(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.markpen_id = np.random.randint(0, 6)
        self.pencup_id = int(np.random.choice([1, 5]))
        side = float(np.random.choice([-1, 1]))
        markpen_x = side * np.random.uniform(0.20, 0.25)
        markpen_y = np.random.uniform(-0.01, 0.05)

        self.markpen = create_actor(
            scene=self,
            pose=sapien.Pose([markpen_x, markpen_y, 0.756], [1, 0, 0, 0]),
            modelname="058_markpen",
            model_id=self.markpen_id,
            convex=True,
        )
        self.pencup = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.1, -0.13, 0.741], [0.5, 0.5, 0.5, 0.5]),
            modelname="059_pencup",
            model_id=self.pencup_id,
            convex=True,
            is_static=True,
        )
        self.markpen.set_mass(0.01)
        self.arm_tag = ArmTag("left" if side < 0 else "right")
        self.add_prohibit_area(self.markpen, padding=0.07)
        self.add_prohibit_area(self.pencup, padding=0.07)
        self.insert_point_id = self._get_insert_point_id()
        self.task_failure_reason = None

    def play_once(self):
        self.set_subtask(0)
        grasp_point_id = 3 if self.arm_tag == "left" else 5
        self.run_action_stage(
            "grasp_markpen",
            lambda: self.grasp_actor(
                self.markpen,
                arm_tag=self.arm_tag,
                contact_point_id=grasp_point_id,
                pre_grasp_dis=0.10,
            )
        )
        lift_x = -0.04 if self.arm_tag == "left" else 0.04
        self.run_action_stage(
            "lift_markpen",
            lambda: self.move_by_displacement(
                self.arm_tag,
                x=lift_x,
                z=0.07,
            ),
        )

        cup_pose = self.pencup.get_pose().p
        target_pose = sapien.Pose(
            [cup_pose[0], cup_pose[1], self._cup_rim_height() - 0.025],
            [0.7071068, 0.7071068, 0, 0],
        )
        self.run_action_stage(
            "align_and_insert_markpen",
            lambda: self.place_actor(
                self.markpen,
                arm_tag=self.arm_tag,
                target_pose=target_pose,
                functional_point_id=self.insert_point_id,
                pre_dis=0.12,
                dis=0.005,
                pre_dis_axis=[0, -1, 0],
                constrain="align",
            )
        )
        self.run_action_stage(
            "retreat_above_pencup",
            lambda: self.move_by_displacement(self.arm_tag, z=0.08),
        )
        self.info["info"] = {
            "{A}": f"058_markpen/base{self.markpen_id}",
            "{B}": f"059_pencup/base{self.pencup_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def _get_insert_point_id(self):
        root_position = self.markpen.get_pose().p
        functional_points = [
            self.markpen.get_functional_point(index, "pose").p
            for index in range(len(self.markpen.config["functional_matrix"]))
        ]
        distances = [
            np.linalg.norm(point - root_position) for point in functional_points
        ]
        return int(np.argmin(distances))

    def _cup_rim_height(self):
        config = self.pencup.config
        center = np.asarray(config["center"], dtype=np.float64)
        half_extents = np.asarray(config["extents"], dtype=np.float64) / 2
        scale = np.asarray(config["scale"], dtype=np.float64)
        local_top = (center + np.array([0, half_extents[1], 0])) * scale
        cup_matrix = self.pencup.get_pose().to_transformation_matrix()
        return float((cup_matrix @ np.append(local_top, 1.0))[2])

    def check_success(self):
        pen_pose = self.markpen.get_pose()
        cup_pose = self.pencup.get_pose().p
        insert_point = self.markpen.get_functional_point(
            self.insert_point_id, "pose"
        ).p
        rim_height = self._cup_rim_height()
        pen_axis = pen_pose.to_transformation_matrix()[:3, :3] @ np.array([0, 1, 0])
        vertical_alignment = float(np.dot(pen_axis, [0, 0, 1]))
        xy_distance = float(np.linalg.norm(insert_point[:2] - cup_pose[:2]))
        upright = vertical_alignment > 0.8
        inside_xy = xy_distance < 0.03
        insertion_depth_ok = rim_height - 0.05 < insert_point[2] < rim_height + 0.01
        actors_in_contact = self.check_actors_contact("058_markpen", "059_pencup")
        left_gripper_open = self.is_left_gripper_open()
        right_gripper_open = self.is_right_gripper_open()

        failure_reasons = []
        if not inside_xy:
            failure_reasons.append(
                f"insert point is outside pencup opening "
                f"(xy_distance={xy_distance:.4f}m, required<0.0300m)"
            )
        if not insertion_depth_ok:
            failure_reasons.append(
                f"insert point depth is invalid "
                f"(z={insert_point[2]:.4f}m, required "
                f"{rim_height - 0.05:.4f}m<z<{rim_height + 0.01:.4f}m)"
            )
        if not upright:
            failure_reasons.append(
                f"markpen is not upright "
                f"(vertical_alignment={vertical_alignment:.4f}, required>0.8000)"
            )
        if not actors_in_contact:
            failure_reasons.append("markpen is not in contact with pencup")
        if not left_gripper_open:
            failure_reasons.append("left gripper is not open")
        if not right_gripper_open:
            failure_reasons.append("right gripper is not open")

        self.task_failure_reason = "; ".join(failure_reasons) or None
        return not failure_reasons
