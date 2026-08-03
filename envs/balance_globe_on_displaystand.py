import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


class balance_globe_on_displaystand(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        side = float(np.random.choice([-1, 1]))
        self.globe_id = int(np.random.choice([2, 3]))
        self.displaystand_id = np.random.randint(0, 5)
        self.globe = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.23, 0.02, 0.741], [0.5, 0.5, 0.5, 0.5]),
            modelname="089_globe",
            model_id=self.globe_id,
            convex=True,
        )
        self.displaystand = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.06, -0.14, 0.741], [0.5, 0.5, 0.5, 0.5]),
            modelname="074_displaystand",
            model_id=self.displaystand_id,
            convex=True,
            is_static=True,
        )
        self.globe.set_mass(0.02)
        self.arm_tag = ArmTag("left" if side < 0 else "right")
        self.add_prohibit_area(self.globe, padding=0.09)
        self.add_prohibit_area(self.displaystand, padding=0.07)

    def play_once(self):
        self.set_subtask(0)
        self.move(self.grasp_actor(self.globe, arm_tag=self.arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(self.arm_tag, z=0.12))
        stand_fp = self.displaystand.get_functional_point(0, "pose").p
        target_pose = sapien.Pose(
            [stand_fp[0], stand_fp[1], stand_fp[2] + 0.025],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.move(
            self.place_actor(
                self.globe,
                arm_tag=self.arm_tag,
                target_pose=target_pose,
                pre_dis=0.1,
                dis=0.005,
                constrain="align",
            )
        )
        self.move(self.move_by_displacement(self.arm_tag, z=0.08))
        self.info["info"] = {
            "{A}": f"089_globe/base{self.globe_id}",
            "{B}": f"074_displaystand/base{self.displaystand_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        globe_pose = self.globe.get_pose()
        stand_fp = self.displaystand.get_functional_point(0, "pose").p
        globe_axis = globe_pose.to_transformation_matrix()[:3, :3] @ np.array([0, 1, 0])
        return (
            np.linalg.norm(globe_pose.p[:2] - stand_fp[:2]) < 0.04
            and globe_pose.p[2] > self.displaystand.get_pose().p[2] + 0.03
            and np.dot(globe_axis, [0, 0, 1]) > 0.8
            and self.check_actors_contact("089_globe", "074_displaystand")
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
