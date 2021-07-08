"""格式化输出模型的汇总信息，包括：模型结构、参数量、大小(参数和buffers的占用空间)以及FLOPs"""

from torchscan import crawl_module
from torchscan.utils import aggregate_info, format_info


def get_module_sum_info(module, input_shape, wrap_mode='mid', max_depth=None):
    """Get module summary for an expected input tensor shape

        Example::
            >>> import torch.nn as nn
            >>> from torchscan import summary
            >>> mod = nn.Conv2d(3, 8, 3)
            >>> summary(mod, (3, 224, 224))

        Args:
            module (torch.nn.Module): module to inspect
            input_shape (tuple<int>): expected input shapes
            wrap_mode (str, optional): if a value is too long, where the wrapping should be performed
            max_depth (int, optional): maximum depth of layer information
    """

    # Get the summary dict
    module_info = crawl_module(module, input_shape)
    # Aggregate until max_depth
    if isinstance(max_depth, int):
        module_info = aggregate_info(module_info, max_depth)

    # Format it
    return format_info(module_info, wrap_mode=wrap_mode)
