from ._household_task_policy import SwapHouseholdObjectsPolicy


class swap_phone_and_remotecontrol(SwapHouseholdObjectsPolicy):
    object_a_spec = ("077_phone", [0, 1, 2, 4])
    object_b_spec = ("079_remotecontrol", [0, 1, 2, 3, 4, 5, 6])
