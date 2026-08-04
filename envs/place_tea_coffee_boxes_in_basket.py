from ._household_task_policy import PlaceHouseholdPairPolicy


class place_tea_coffee_boxes_in_basket(PlaceHouseholdPairPolicy):
    source_specs = (
        ("112_tea-box", [0, 1, 2, 3, 4, 5]),
        ("113_coffee-box", [0, 1, 2, 3, 4, 5, 6]),
    )
    target_spec = ("110_basket", [0, 1, 2, 3])
    target_ids = (0, 1)
    target_threshold = 0.065
