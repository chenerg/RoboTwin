import numpy as np

from ._base_task import Base_Task
from .utils import *


class open_then_close_cabinet_drawer(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs, table_static=False)

    def load_actors(self):
        self.model_name = "036_cabinet"
        self.model_id = 46653
        self.cabinet = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.03, 0.03],
            ylim=[0.145, 0.165],
            rotate_rand=False,
            qpos=[1, 0, 0, 1],
            fix_root_link=True,
        )
        limit = self.cabinet.get_qlimits()[0]
        qpos = np.asarray(self.cabinet.get_qpos(), dtype=np.float64).copy()
        qpos[0] = limit[0] + (limit[1] - limit[0]) * 0.05
        self.cabinet.set_qpos(qpos)
        self.cabinet.set_mass(0.01)
        self.cabinet.set_properties(1, 0)
        self.arm_tag = ArmTag("right")
        self.add_prohibit_area(self.cabinet, padding=0.12)
        self.prohibited_area.append([-0.15, -0.25, 0.15, 0.3])

    def play_once(self):
        self.set_subtask(0)
        self.run_action_stage(
            "grasp_drawer_handle",
            lambda: self.grasp_actor(
                self.cabinet,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.06,
                contact_point_id=0,
            )
        )
        for step in range(5):
            self.run_action_stage(
                f"open_drawer_step_{step + 1}",
                lambda: self.move_by_displacement(self.arm_tag, y=-0.04),
            )
            self.check_success()

        self.set_subtask(1)
        for step in range(5):
            self.run_action_stage(
                f"close_drawer_step_{step + 1}",
                lambda: self.move_by_displacement(self.arm_tag, y=0.04),
            )

        self.run_action_stage(
            "release_drawer_handle",
            lambda: self.open_gripper(self.arm_tag),
        )
        self.run_action_stage(
            "return_arm_to_origin",
            lambda: self.back_to_origin(self.arm_tag),
        )
        self.info["info"] = {
            "{A}": "036_cabinet/base0",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def _drawer_fraction(self):
        limit = self.cabinet.get_qlimits()[0]
        span = max(float(limit[1] - limit[0]), 1e-6)
        return float((self.cabinet.get_qpos()[0] - limit[0]) / span)

    def check_success(self):
        if self._drawer_fraction() >= 0.35:
            self.stage_success_tag = True
        return (
            self.stage_success_tag
            and self._drawer_fraction() <= 0.2
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
