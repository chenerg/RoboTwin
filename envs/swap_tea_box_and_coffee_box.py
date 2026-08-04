from ._household_task_policy import SwapHouseholdObjectsPolicy


class swap_tea_box_and_coffee_box(SwapHouseholdObjectsPolicy):
    object_a_spec = ("112_tea-box", [0, 1, 2, 3, 4, 5])
    object_b_spec = ("113_coffee-box", [0, 1, 2, 3, 4, 5, 6])
