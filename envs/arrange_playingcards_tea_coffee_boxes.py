from ._household_task_policy import ArrangeHouseholdObjectsPolicy


class arrange_playingcards_tea_coffee_boxes(ArrangeHouseholdObjectsPolicy):
    object_specs = (
        ("081_playingcards", [0, 1, 2]),
        ("112_tea-box", [0, 1, 2, 3, 4, 5]),
        ("113_coffee-box", [0, 1, 2, 3, 4, 5, 6]),
    )
