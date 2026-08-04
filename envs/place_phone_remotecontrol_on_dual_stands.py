from ._household_task_policy import PlaceHouseholdPairPolicy


class place_phone_remotecontrol_on_dual_stands(PlaceHouseholdPairPolicy):
    source_specs = (
        ("077_phone", [0, 1, 2, 4]),
        ("079_remotecontrol", [0, 1, 2, 3, 4, 5, 6]),
    )
    source_functional_points = (0, 0)
    target_spec = ("074_displaystand", [0, 1, 2, 3, 4])
    target_ids = (0, 0)
    separate_targets = True
    target_quaternion = (0.7071068, 0.7071068, 0, 0)
