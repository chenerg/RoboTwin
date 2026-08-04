from ._household_task_policy import ArrangeHouseholdObjectsPolicy


class arrange_bread_soap_rubikscube(ArrangeHouseholdObjectsPolicy):
    object_specs = (
        ("075_bread", [0, 1, 2, 3, 4, 5, 6]),
        ("107_soap", [0, 1, 2, 3]),
        ("073_rubikscube", [0, 1, 2]),
    )
