import numpy as np
import sapien

from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from ._base_task import Base_Task
from .utils import *


class push_toycar_to_parking_zone(Base_Task):

    BEHIND_CAR_OFFSET = 0.065
    PUSH_HEIGHT_OFFSET = 0.05
    SAFE_APPROACH_HEIGHT_OFFSET = 0.14
    PARKING_END_OFFSET = 0.045
    PUSH_GRIPPER_POS = 0.3

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        side = float(np.random.choice([-1, 1]))
        self.toycar_id = np.random.randint(0, 6)
        lane_x = side * np.random.uniform(0.15, 0.21)
        toycar_y = np.random.uniform(0.01, 0.05)
        self.toycar = create_actor(
            scene=self,
            pose=sapien.Pose([lane_x, toycar_y, 0.741], [0.7071068, 0.7071068, 0, 0]),
            modelname="057_toycar",
            model_id=self.toycar_id,
            convex=True,
        )
        self.parking_pad = create_box(
            scene=self,
            pose=sapien.Pose([lane_x, -0.17, 0.741], [1, 0, 0, 0]),
            half_size=(0.065, 0.06, 0.0005),
            color=(0.15, 0.45, 1.0),
            is_static=True,
            name="parking_zone",
        )
        self.toycar.set_mass(0.02)
        self.arm_tag = ArmTag("left" if side < 0 else "right")
        self.push_contacted = False
        self.add_prohibit_area(self.toycar, padding=0.07)
        self.add_prohibit_area(self.parking_pad, padding=0.06)

    def play_once(self):
        self.set_subtask(0)
        car_p = self.toycar.get_pose().p
        push_y = car_p[1] + self.BEHIND_CAR_OFFSET
        push_z = car_p[2] + self.PUSH_HEIGHT_OFFSET
        pre_push_above = [
            car_p[0],
            push_y,
            car_p[2] + self.SAFE_APPROACH_HEIGHT_OFFSET,
        ] + GRASP_DIRECTION_DIC["front"]
        pre_push = [car_p[0], push_y, push_z] + GRASP_DIRECTION_DIC["front"]
        target_p = self.parking_pad.get_pose().p
        push_end = [
            target_p[0],
            target_p[1] + self.PARKING_END_OFFSET,
            push_z,
        ] + GRASP_DIRECTION_DIC["front"]

        self.run_action_stage(
            "set_gripper_for_push",
            lambda: self.close_gripper(self.arm_tag, pos=self.PUSH_GRIPPER_POS),
        )
        self.run_action_stage(
            "move_above_behind_toycar",
            lambda: self.move_to_pose(self.arm_tag, pre_push_above),
        )
        self.run_action_stage(
            "move_behind_toycar",
            lambda: self.move_to_pose(self.arm_tag, pre_push),
        )
        self.run_action_stage(
            "push_toycar_to_parking_zone",
            lambda: self.move_to_pose(self.arm_tag, push_end),
        )
        self.check_success()
        self.run_action_stage(
            "retreat_above_toycar",
            lambda: self.move_by_displacement(self.arm_tag, z=0.08),
        )
        self.run_action_stage(
            "open_gripper_after_push",
            lambda: self.open_gripper(self.arm_tag),
        )
        self.info["info"] = {
            "{A}": f"057_toycar/base{self.toycar_id}",
            "{B}": "blue parking zone",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        if self.get_gripper_actor_contact_position("057_toycar"):
            self.push_contacted = True
        car_p = self.toycar.get_pose().p
        pad_p = self.parking_pad.get_pose().p
        return (
            self.push_contacted
            and abs(car_p[0] - pad_p[0]) < 0.05
            and abs(car_p[1] - pad_p[1]) < 0.05
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
