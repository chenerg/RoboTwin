from ._primitive_task_policy import RelativeBlockPlacementPolicy


class place_red_block_right_of_blue_block(RelativeBlockPlacementPolicy):
    relation = "right of"
