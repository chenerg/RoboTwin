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
