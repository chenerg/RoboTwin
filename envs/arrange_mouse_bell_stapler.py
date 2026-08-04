from ._household_task_policy import ArrangeHouseholdObjectsPolicy


class arrange_mouse_bell_stapler(ArrangeHouseholdObjectsPolicy):
    object_specs = (
        ("047_mouse", [0, 1, 2]),
        ("050_bell", [0, 1]),
        ("048_stapler", [0, 1, 2, 3, 4, 5, 6]),
    )
