from ._household_task_policy import SwapHouseholdObjectsPolicy


class swap_bell_and_rubikscube(SwapHouseholdObjectsPolicy):
    object_a_spec = ("050_bell", [0, 1])
    object_b_spec = ("073_rubikscube", [0, 1, 2])
