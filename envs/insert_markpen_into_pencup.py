import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


class insert_markpen_into_pencup(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.markpen_id = np.random.randint(0, 6)
        self.pencup_id = np.random.randint(0, 7)
        side = float(np.random.choice([-1, 1]))

        self.markpen = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.23, 0.02, 0.756], [1, 0, 0, 0]),
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

    def play_once(self):
        self.set_subtask(0)
        self.move(
            self.grasp_actor(
                self.markpen,
                arm_tag=self.arm_tag,
                contact_point_id=[2, 4, 6],
                pre_grasp_dis=0.09,
            )
        )
        self.move(self.move_by_displacement(self.arm_tag, z=0.12))

        cup_pose = self.pencup.get_pose().p
        target_pose = sapien.Pose(
            [cup_pose[0], cup_pose[1], self._cup_rim_height() - 0.025],
            [0.7071068, 0.7071068, 0, 0],
        )
        self.move(
            self.place_actor(
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
        self.move(self.move_by_displacement(self.arm_tag, z=0.08))
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
        upright = np.dot(pen_axis, [0, 0, 1]) > 0.8
        inside_xy = np.linalg.norm(insert_point[:2] - cup_pose[:2]) < 0.03
        insertion_depth_ok = rim_height - 0.05 < insert_point[2] < rim_height + 0.01
        return (
            inside_xy
            and insertion_depth_ok
            and upright
            and self.check_actors_contact("058_markpen", "059_pencup")
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
