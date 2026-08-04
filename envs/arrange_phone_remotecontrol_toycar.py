from ._household_task_policy import ArrangeHouseholdObjectsPolicy


class arrange_phone_remotecontrol_toycar(ArrangeHouseholdObjectsPolicy):
    object_specs = (
        ("077_phone", [0, 1, 2, 4]),
        ("079_remotecontrol", [0, 1, 2, 3, 4, 5, 6]),
        ("057_toycar", [0, 1, 2, 3, 4, 5]),
    )
