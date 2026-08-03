from ._primitive_task_policy import StackBlocksPolicy


class stack_green_block_on_yellow_block(StackBlocksPolicy):
    top_color = "green"
    bottom_color = "yellow"
