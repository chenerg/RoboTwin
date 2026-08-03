import numpy as np

from ._base_task import Base_Task
from .utils import *


class close_laptop(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.model_name = "015_laptop"
        self.model_id = np.random.randint(0, 11)
        self.laptop = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[-0.1, 0.05],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 3],
            qpos=[0.7, 0, 0, 0.7],
            fix_root_link=True,
        )
        limit = self.laptop.get_qlimits()[0]
        self.laptop.set_qpos([limit[0] + (limit[1] - limit[0]) * 0.8])
        self.laptop.set_mass(0.01)
        self.laptop.set_properties(1, 0)
        self.add_prohibit_area(self.laptop, padding=0.1)

    def play_once(self):
        face_prod = get_face_prod(self.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        self.arm_tag = ArmTag("left" if face_prod > 0 else "right")
        self.set_subtask(0)
        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{a}": str(self.arm_tag),
        }

        # Reverse open_laptop's proven CP0 -> CP1 trajectory: grasp the
        # opened screen at CP1, then progressively move it toward CP0.
        self.move(
            self.grasp_actor(
                self.laptop,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.08,
                contact_point_id=1,
            )
        )
        if not self.plan_success:
            return self.info

        for _ in range(8):
            self.move(
                self.grasp_actor(
                    self.laptop,
                    arm_tag=self.arm_tag,
                    pre_grasp_dis=0.0,
                    grasp_dis=0.0,
                    contact_point_id=0,
                )
            )
            if not self.plan_success or self._closed_fraction() <= 0.25:
                break

        if self.plan_success:
            self.move(self.open_gripper(self.arm_tag))
            self.move(self.back_to_origin(self.arm_tag))
        return self.info

    def _closed_fraction(self):
        limit = self.laptop.get_qlimits()[0]
        span = max(float(limit[1] - limit[0]), 1e-6)
        return float((self.laptop.get_qpos()[0] - limit[0]) / span)

    def check_success(self):
        if not hasattr(self, "arm_tag"):
            return False
        return self._closed_fraction() <= 0.3 and self.is_left_gripper_open() and self.is_right_gripper_open()
