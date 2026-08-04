from ._household_task_policy import PlaceHouseholdPairPolicy


class place_bread_can_on_tray(PlaceHouseholdPairPolicy):
    source_specs = (
        ("075_bread", [0, 1, 2, 3, 4, 5, 6]),
        ("071_can", [0, 1, 2, 3, 5, 6]),
    )
    target_spec = ("008_tray", [0, 1, 2, 3])
    target_ids = (0, 1)
    target_quaternion = (0.706527, 0.706483, -0.0291356, -0.0291767)
    target_scale = (2.0, 2.0, 2.0)
    target_threshold = 0.08
