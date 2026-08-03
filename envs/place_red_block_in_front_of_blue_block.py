from ._primitive_task_policy import RelativeBlockPlacementPolicy


class place_red_block_in_front_of_blue_block(RelativeBlockPlacementPolicy):
    relation = "in front of"
