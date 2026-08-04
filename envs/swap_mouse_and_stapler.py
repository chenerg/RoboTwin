from ._household_task_policy import SwapHouseholdObjectsPolicy


class swap_mouse_and_stapler(SwapHouseholdObjectsPolicy):
    object_a_spec = ("047_mouse", [0, 1, 2])
    object_b_spec = ("048_stapler", [0, 1, 2, 3, 4, 5, 6])
