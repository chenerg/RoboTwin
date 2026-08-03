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
            pose=sapien.Pose([side * 0.06, -0.13, 0.741], [0.5, 0.5, 0.5, 0.5]),
            modelname="059_pencup",
            model_id=self.pencup_id,
            convex=True,
            is_static=True,
        )
        self.markpen.set_mass(0.01)
        self.arm_tag = ArmTag("left" if side < 0 else "right")
        self.add_prohibit_area(self.markpen, padding=0.07)
        self.add_prohibit_area(self.pencup, padding=0.07)

    def play_once(self):
        self.set_subtask(0)
        self.move(
            self.grasp_actor(
                self.markpen,
                arm_tag=self.arm_tag,
                contact_point_id=[0, 2, 4, 6],
                pre_grasp_dis=0.09,
            )
        )
        self.move(self.move_by_displacement(self.arm_tag, z=0.12))

        cup_pose = self.pencup.get_pose().p
        target_pose = sapien.Pose(
            [cup_pose[0], cup_pose[1], cup_pose[2] + 0.105],
            [0.7071068, 0.7071068, 0, 0],
        )
        self.move(
            self.place_actor(
                self.markpen,
                arm_tag=self.arm_tag,
                target_pose=target_pose,
                pre_dis=0.1,
                dis=0.015,
                pre_dis_axis=[0, 0, 1],
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

    def check_success(self):
        pen_pose = self.markpen.get_pose()
        cup_pose = self.pencup.get_pose().p
        pen_axis = pen_pose.to_transformation_matrix()[:3, :3] @ np.array([0, 1, 0])
        upright = np.dot(pen_axis, [0, 0, 1]) > 0.8
        inside_xy = np.linalg.norm(pen_pose.p[:2] - cup_pose[:2]) < 0.035
        height_ok = cup_pose[2] + 0.055 < pen_pose.p[2] < cup_pose[2] + 0.17
        return (
            inside_xy
            and height_ok
            and upright
            and self.check_actors_contact("058_markpen", "059_pencup")
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
