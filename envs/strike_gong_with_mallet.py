import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


class strike_gong_with_mallet(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        side = float(np.random.choice([-1, 1]))
        self.mallet_id = 3
        self.gong_id = np.random.randint(0, 6)
        self.mallet = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.23, 0.04, 0.77], [1, 0, 0, 0]),
            modelname="084_woodenmallet",
            model_id=self.mallet_id,
            convex=True,
        )
        self.gong = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.08, -0.11, 0.741], [0.7071068, 0.7071068, 0, 0]),
            modelname="085_gong",
            model_id=self.gong_id,
            convex=True,
            is_static=True,
        )
        self.mallet.set_mass(0.01)
        self.arm_tag = ArmTag("left" if side < 0 else "right")
        self.add_prohibit_area(self.mallet, padding=0.08)
        self.add_prohibit_area(self.gong, padding=0.09)

    def play_once(self):
        self.set_subtask(0)
        self.run_action_stage(
            "grasp_mallet",
            lambda: self.grasp_actor(
                self.mallet,
                arm_tag=self.arm_tag,
                contact_point_id=[0, 1],
                pre_grasp_dis=0.1,
            )
        )
        self.run_action_stage(
            "lift_mallet",
            lambda: self.move_by_displacement(self.arm_tag, z=0.1),
        )
        self.run_action_stage(
            "strike_gong",
            lambda: self.place_actor(
                self.mallet,
                arm_tag=self.arm_tag,
                target_pose=self.gong.get_functional_point(2),
                functional_point_id=0,
                pre_dis=0.1,
                dis=0.0,
                is_open=False,
                pre_dis_axis="fp",
                constrain="align",
            )
        )
        self.check_success()
        self.run_action_stage(
            "retract_mallet",
            lambda: self.move_by_displacement(
                self.arm_tag,
                z=0.08,
                move_axis="arm",
            ),
        )
        self.info["info"] = {
            "{A}": f"084_woodenmallet/base{self.mallet_id}",
            "{B}": f"085_gong/base{self.gong_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        if self.stage_success_tag:
            return True
        mallet_fp = self.mallet.get_functional_point(0, "pose").p
        gong_fp = self.gong.get_functional_point(2, "pose").p
        if (
            np.linalg.norm(mallet_fp - gong_fp) < 0.045
            and self.check_actors_contact("084_woodenmallet", "085_gong")
        ):
            self.stage_success_tag = True
        return self.stage_success_tag
