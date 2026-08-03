import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


class handover_dumbbell(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        side = float(np.random.choice([-1, 1]))
        self.dumbbell_id = int(np.random.choice([0, 2, 4, 6]))
        self.dumbbell = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.21, 0.0, 0.77], [0.7071068, 0.7071068, 0, 0]),
            modelname="052_dumbbell",
            model_id=self.dumbbell_id,
            convex=True,
        )
        self.dumbbell.set_mass(0.03)
        self.giver_arm = ArmTag("left" if side < 0 else "right")
        self.receiver_arm = self.giver_arm.opposite
        self.add_prohibit_area(self.dumbbell, padding=0.09)

    def play_once(self):
        self.set_subtask(0)
        self.move(
            self.grasp_actor(
                self.dumbbell,
                arm_tag=self.giver_arm,
                contact_point_id=0,
                pre_grasp_dis=0.1,
            )
        )
        self.move(self.move_by_displacement(self.giver_arm, z=0.12))
        self.move(
            self.place_actor(
                self.dumbbell,
                arm_tag=self.giver_arm,
                target_pose=[0, -0.05, 0.95, 0.7071068, 0.7071068, 0, 0],
                pre_dis=0.02,
                dis=0.0,
                is_open=False,
                constrain="free",
            )
        )

        self.set_subtask(1)
        self.move(
            self.grasp_actor(
                self.dumbbell,
                arm_tag=self.receiver_arm,
                contact_point_id=1,
                pre_grasp_dis=0.1,
            )
        )
        self.move(self.open_gripper(self.giver_arm))
        self.move(
            self.move_by_displacement(self.giver_arm, z=0.07),
            self.move_by_displacement(
                self.receiver_arm,
                x=-0.07 if self.receiver_arm == "left" else 0.07,
            ),
        )
        self.info["info"] = {
            "{A}": f"052_dumbbell/base{self.dumbbell_id}",
            "{a}": str(self.giver_arm),
            "{b}": str(self.receiver_arm),
        }
        return self.info

    def check_success(self):
        if not hasattr(self, "receiver_arm"):
            return False
        p = self.dumbbell.get_pose().p
        receiver_closed = self.is_left_gripper_close if self.receiver_arm == "left" else self.is_right_gripper_close
        giver_open = self.is_left_gripper_open if self.giver_arm == "left" else self.is_right_gripper_open
        in_receiver_side = p[0] < 0 if self.receiver_arm == "left" else p[0] > 0
        return receiver_closed() and giver_open() and in_receiver_side and p[2] > 0.87
