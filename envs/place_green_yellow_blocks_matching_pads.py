from ._primitive_task_policy import PlaceBlocksOnPadsPolicy


class place_green_yellow_blocks_matching_pads(PlaceBlocksOnPadsPolicy):
    block_colors = ("green", "yellow")
    pad_colors = ("green", "yellow")
    target_indices = (0, 1)
