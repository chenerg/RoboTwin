from ._household_task_policy import SwapHouseholdObjectsPolicy


class swap_mouse_and_stapler(SwapHouseholdObjectsPolicy):
    object_a_spec = ("047_mouse", [0, 1, 2])
    object_b_spec = ("048_stapler", [0, 1, 2, 3, 4, 5, 6])

    # Match place_mouse_pad for both mouse handoffs.  The mouse has no
    # functional point, so both placements still align its actor root.
    object_a_grasp_pre_dis = 0.1
    object_a_lift_height = 0.1
    object_a_place_pre_dis = 0.07
    object_a_place_dis = 0.005
    object_a_place_constrain = "align"
    object_a_target_quaternion = (0, 0, 0, 1)

    def _relocate_object_b(self, arm_tag, target_pose, stage_prefix, *, return_to_origin=False):
        # Move the stapler exactly like move_stapler_pad: a direct grasp,
        # a lift along the gripper's local approach axis, and an aligned
        # flush placement, instead of the staged cycle used by the shared
        # swap policy.
        self.move(self.grasp_actor(self.object_b, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))
        self.move(
            self.place_actor(
                self.object_b,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=0.1,
                dis=0.0,
                constrain="align",
            ))
        self.move(self.move_by_displacement(arm_tag, z=0.08))
        if return_to_origin:
            self.move(self.back_to_origin(arm_tag))
