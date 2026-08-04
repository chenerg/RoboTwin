from ._household_task_policy import PlaceHouseholdPairPolicy


class place_playingcards_toycar_in_plasticbox(PlaceHouseholdPairPolicy):
    source_specs = (
        ("081_playingcards", [0, 1, 2]),
        ("057_toycar", [0, 1, 2, 3, 4, 5]),
    )
    target_spec = ("062_plasticbox", [3, 5])
    target_ids = (1, 0)
    target_threshold = 0.06
