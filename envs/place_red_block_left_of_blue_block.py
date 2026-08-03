from ._primitive_task_policy import RelativeBlockPlacementPolicy


class place_red_block_left_of_blue_block(RelativeBlockPlacementPolicy):
    relation = "left of"
