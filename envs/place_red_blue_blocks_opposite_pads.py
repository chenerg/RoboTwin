from ._primitive_task_policy import PlaceBlocksOnPadsPolicy


class place_red_blue_blocks_opposite_pads(PlaceBlocksOnPadsPolicy):
    block_colors = ("red", "blue")
    pad_colors = ("red", "blue")
    target_indices = (1, 0)
