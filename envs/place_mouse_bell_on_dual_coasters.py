from ._household_task_policy import PlaceHouseholdPairPolicy


class place_mouse_bell_on_dual_coasters(PlaceHouseholdPairPolicy):
    source_specs = (
        ("047_mouse", [0, 1, 2]),
        ("050_bell", [0, 1]),
    )
    target_spec = ("019_coaster", [0])
    target_ids = (0, 0)
    separate_targets = True
