import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


class weigh_then_remove_object(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        side = float(np.random.choice([-1, 1]))
        self.object_name = "047_mouse"
        self.object_id = np.random.randint(0, 3)
        self.scale_id = int(np.random.choice([0, 1, 5, 6]))
        self.object = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.23, 0.04, 0.741], [0.5, 0.5, 0.5, 0.5]),
            modelname=self.object_name,
            model_id=self.object_id,
            convex=True,
        )
        self.scale = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.07, -0.07, 0.741], [0.5, 0.5, 0.5, 0.5]),
            modelname="072_electronicscale",
            model_id=self.scale_id,
            convex=True,
            is_static=True,
        )
        self.target_pad = create_box(
            scene=self,
            pose=sapien.Pose([side * 0.18, -0.2, 0.741], [1, 0, 0, 0]),
            half_size=(0.055, 0.055, 0.0005),
            color=(0.0, 0.2, 1.0),
            is_static=True,
            name="weigh_target_pad",
        )
        self.object.set_mass(0.03)
        self.arm_tag = ArmTag("left" if side < 0 else "right")
        self.scale_stage_complete = False
        self.add_prohibit_area(self.object, padding=0.06)
        self.add_prohibit_area(self.scale, padding=0.07)
        self.add_prohibit_area(self.target_pad, padding=0.06)

    def play_once(self):
        self.set_subtask(0)
        self.run_action_stage(
            "grasp_object_for_weighing",
            lambda: self.grasp_actor(
                self.object,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.09,
            ),
        )
        self.run_action_stage(
            "lift_object_for_weighing",
            lambda: self.move_by_displacement(self.arm_tag, z=0.12),
        )
        self.run_action_stage(
            "place_object_on_scale",
            lambda: self.place_actor(
                self.object,
                arm_tag=self.arm_tag,
                target_pose=self.scale.get_functional_point(0),
                pre_dis=0.07,
                dis=0.005,
                constrain="free",
            )
        )
        self.check_success()
        self.run_action_stage(
            "retreat_after_weighing",
            lambda: self.move_by_displacement(self.arm_tag, z=0.08),
        )

        self.set_subtask(1)
        self.run_action_stage(
            "grasp_object_from_scale",
            lambda: self.grasp_actor(
                self.object,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.09,
            ),
        )
        self.run_action_stage(
            "lift_object_from_scale",
            lambda: self.move_by_displacement(self.arm_tag, z=0.1),
        )
        self.run_action_stage(
            "place_object_on_target_pad",
            lambda: self.place_actor(
                self.object,
                arm_tag=self.arm_tag,
                target_pose=self.target_pad.get_functional_point(1),
                pre_dis=0.07,
                dis=0.005,
                constrain="free",
            )
        )
        self.run_action_stage(
            "retreat_after_target_placement",
            lambda: self.move_by_displacement(self.arm_tag, z=0.07),
        )
        self.info["info"] = {
            "{A}": f"{self.object_name}/base{self.object_id}",
            "{B}": f"072_electronicscale/base{self.scale_id}",
            "{C}": "blue pad",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        object_p = self.object.get_pose().p
        scale_fp = self.scale.get_functional_point(0, "pose").p
        if (
            np.linalg.norm(object_p[:2] - scale_fp[:2]) < 0.04
            and object_p[2] > scale_fp[2] - 0.01
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        ):
            self.scale_stage_complete = True

        pad_fp = self.target_pad.get_functional_point(1, "pose").p
        on_pad = np.linalg.norm(object_p[:2] - pad_fp[:2]) < 0.04 and object_p[2] > pad_fp[2] - 0.01
        return self.scale_stage_complete and on_pad and self.is_left_gripper_open() and self.is_right_gripper_open()
