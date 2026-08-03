from ._primitive_task_policy import RelativeBlockPlacementPolicy


class place_red_block_behind_blue_block(RelativeBlockPlacementPolicy):
    relation = "behind"
