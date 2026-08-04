import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import *


class pour_beads_between_bowls(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        side = float(np.random.choice([-1, 1]))
        source_x = side * np.random.uniform(0.19, 0.23)
        source_y = np.random.uniform(-0.03, 0.01)
        source_pose = sapien.Pose([source_x, source_y, 0.741], [0.5, 0.5, 0.5, 0.5])
        target_pose = sapien.Pose([side * 0.05, -0.14, 0.741], [0.5, 0.5, 0.5, 0.5])
        self.source_bowl = create_actor(
            scene=self,
            pose=source_pose,
            modelname="002_bowl",
            model_id=3,
            convex=True,
        )
        self.target_bowl = create_actor(
            scene=self,
            pose=target_pose,
            modelname="002_bowl",
            model_id=3,
            convex=True,
            is_static=True,
        )
        self.source_bowl.set_name("source_bowl")
        self.target_bowl.set_name("target_bowl")
        self.source_bowl.set_mass(0.03)
        self.arm_tag = ArmTag("left" if side < 0 else "right")

        self.beads = []
        for index in range(5):
            bead = create_sphere(
                scene=self,
                pose=sapien.Pose(
                    [
                        source_pose.p[0] + np.random.uniform(-0.012, 0.012),
                        source_pose.p[1] + np.random.uniform(-0.012, 0.012),
                        0.785 + index * 0.011,
                    ],
                    [1, 0, 0, 0],
                ),
                radius=0.007,
                color=(1.0, 0.35, 0.0),
                name=f"bead_{index}",
            )
            bead.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.0002
            self.beads.append(bead)

        self.add_prohibit_area(self.source_bowl, padding=0.08)
        self.add_prohibit_area(self.target_bowl, padding=0.08)

    def play_once(self):
        self.set_subtask(0)
        contact_point_id = 0 if self.arm_tag == "right" else 2
        self.run_action_stage(
            "grasp_source_bowl",
            lambda: self.grasp_actor(
                self.source_bowl,
                arm_tag=self.arm_tag,
                contact_point_id=contact_point_id,
                pre_grasp_dis=0.1,
            )
        )
        self.run_action_stage(
            "lift_source_bowl",
            lambda: self.move_by_displacement(self.arm_tag, z=0.13),
        )

        target_p = self.target_bowl.get_pose().p
        tilt = t3d.euler.euler2quat(0, 1.15 if self.arm_tag == "left" else -1.15, 0)
        target_q = t3d.quaternions.qmult(self.source_bowl.get_pose().q, tilt)
        pour_pose = sapien.Pose([target_p[0], target_p[1], target_p[2] + 0.2], target_q)
        self.run_action_stage(
            "move_and_tilt_bowl_above_target",
            lambda: self.place_actor(
                self.source_bowl,
                arm_tag=self.arm_tag,
                target_pose=pour_pose,
                pre_dis=0.08,
                dis=0.0,
                is_open=False,
                constrain="align",
            )
        )
        for step in range(2):
            self.run_action_stage(
                f"shake_bowl_outward_{step + 1}",
                lambda: self.move_by_displacement(
                    self.arm_tag,
                    x=0.02 if self.arm_tag == "left" else -0.02,
                ),
            )
            self.run_action_stage(
                f"shake_bowl_inward_{step + 1}",
                lambda: self.move_by_displacement(
                    self.arm_tag,
                    x=-0.02 if self.arm_tag == "left" else 0.02,
                ),
            )
        self.delay(2)
        self.info["info"] = {
            "{A}": "002_bowl/base3",
            "{B}": "002_bowl/base3",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        target_p = self.target_bowl.get_pose().p
        for bead in self.beads:
            bead_p = bead.get_pose().p
            if np.linalg.norm(bead_p[:2] - target_p[:2]) >= 0.075:
                return False
            if not target_p[2] < bead_p[2] < target_p[2] + 0.14:
                return False
        return self.source_bowl.get_pose().p[2] > target_p[2] + 0.12
