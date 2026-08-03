import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


class wipe_mini_chalkboard(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        side = float(np.random.choice([-1, 1]))
        self.eraser = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.23, 0.04, 0.76], [1, 0, 0, 0]),
            modelname="117_whiteboard-eraser",
            model_id=0,
            convex=True,
        )
        self.chalkboard = create_actor(
            scene=self,
            pose=sapien.Pose([side * 0.07, -0.11, 0.741], [0.7071068, 0.7071068, 0, 0]),
            modelname="119_mini-chalkboard",
            model_id=0,
            convex=True,
            is_static=True,
        )
        self.eraser.set_mass(0.01)
        self.arm_tag = ArmTag("left" if side < 0 else "right")
        self.wipe_zones = set()
        self.add_prohibit_area(self.eraser, padding=0.07)
        self.add_prohibit_area(self.chalkboard, padding=0.09)

    def play_once(self):
        self.set_subtask(0)
        self.move(self.grasp_actor(self.eraser, arm_tag=self.arm_tag, pre_grasp_dis=0.09))
        self.move(
            self.place_actor(
                self.eraser,
                arm_tag=self.arm_tag,
                target_pose=self.chalkboard.get_functional_point(0),
                functional_point_id=0,
                pre_dis=0.08,
                dis=0.005,
                is_open=False,
                pre_dis_axis="fp",
                constrain="align",
            )
        )
        self.move(self.move_by_displacement(self.arm_tag, x=-0.045))
        self.check_success()
        self.move(self.move_by_displacement(self.arm_tag, x=0.09))
        self.check_success()
        self.move(self.move_by_displacement(self.arm_tag, x=-0.045))
        self.check_success()
        self.info["info"] = {
            "{A}": "117_whiteboard-eraser/base0",
            "{B}": "119_mini-chalkboard/base0",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        if self.stage_success_tag:
            return True
        if not hasattr(self, "wipe_zones"):
            return False
        eraser_fp = self.eraser.get_functional_point(0, "pose").p
        board_fp = self.chalkboard.get_functional_point(0, "pose").p
        if self.check_actors_contact("117_whiteboard-eraser", "119_mini-chalkboard"):
            x_offset = eraser_fp[0] - board_fp[0]
            if x_offset < -0.025:
                self.wipe_zones.add("left")
            elif x_offset > 0.025:
                self.wipe_zones.add("right")
            else:
                self.wipe_zones.add("center")
        if self.wipe_zones == {"left", "center", "right"}:
            self.stage_success_tag = True
        return self.stage_success_tag
