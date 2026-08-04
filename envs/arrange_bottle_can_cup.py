from ._household_task_policy import ArrangeHouseholdObjectsPolicy


class arrange_bottle_can_cup(ArrangeHouseholdObjectsPolicy):
    object_specs = (
        ("001_bottle", [13, 16]),
        ("071_can", [0, 1, 2, 3, 5, 6]),
        ("021_cup", [0]),
    )
