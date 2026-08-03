from ._primitive_task_policy import PlaceBlocksOnPadsPolicy


class place_orange_purple_blocks_opposite_pads(PlaceBlocksOnPadsPolicy):
    block_colors = ("orange", "purple")
    pad_colors = ("orange", "purple")
    target_indices = (1, 0)
