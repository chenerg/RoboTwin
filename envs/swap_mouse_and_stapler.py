from ._household_task_policy import SwapHouseholdObjectsPolicy, _asset_info


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

    # Central staging slot: reachable by both arms and well inside the head
    # camera view (the default right-edge slot sits at the FOV boundary).
    staging_xy = (0.0, -0.14)

    def play_once(self):
        # Direct 3-move swap: stage the stapler out of its slot, move the
        # mouse straight into the stapler's original spot, then drop the
        # stapler into the mouse's original spot.  Replaces the shared
        # policy's 5-move bare-table relay.
        self.set_subtask(0)
        self._relocate_object_b(
            self.right_arm,
            self.staging_pose,
            "move_stapler_to_staging",
            return_to_origin=True,
        )

        self.set_subtask(1)
        self._pick_place_root(
            self.object_a,
            self.left_arm,
            self.object_a_goal,
            "move_mouse_to_stapler_spot",
            grasp_pre_dis=self.object_a_grasp_pre_dis,
            lift_height=self.object_a_lift_height,
            place_pre_dis=self.object_a_place_pre_dis,
            place_dis=self.object_a_place_dis,
            place_constrain=self.object_a_place_constrain,
            retreat_height=0.08,
            return_to_origin=True,
        )

        self.set_subtask(2)
        self._relocate_object_b(
            self.right_arm,
            self.object_b_goal,
            "move_stapler_to_mouse_spot",
        )

        self.info["info"] = {
            "{A}": _asset_info(self.object_a_name, self.object_a_id),
            "{B}": _asset_info(self.object_b_name, self.object_b_id),
            "{a}": str(self.left_arm),
            "{b}": str(self.right_arm),
        }
        return self.info

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
