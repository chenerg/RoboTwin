from ._household_task_policy import SwapHouseholdObjectsPolicy


class swap_toycar_and_playingcards(SwapHouseholdObjectsPolicy):
    object_a_spec = ("057_toycar", [0, 1, 2, 3, 4, 5])
    object_b_spec = ("081_playingcards", [0, 1, 2])
