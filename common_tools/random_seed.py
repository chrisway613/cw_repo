"""在编写深度学习相关的pipeline时，为了便于复现实验结果以进行分析，通常需要固定随机种子。
需要注意的是，torch, random, numpy, cuda都要分别单独进行设置。"""

import torch
import random
import numpy as np


def setup_seed(seed=12345):
    """设置随机种子，便于复现实验结果。
       注意，random, numpy, torch, cuda都要分别设置"""

    random.seed(seed)
    np.random.seed(seed)
    # cpu
    torch.manual_seed(seed)
    # gpu
    if torch.cuda.is_available():
        # manual_seed()是仅对当前使用的GPU设置
        # 而manual_seed_all()是对所有GPU设置
        torch.cuda.manual_seed_all(seed)
        # 程序起始就预先搜索各种算子(如卷积)对应的最优算法
        # 在网络结构和输入数据不变时，后续就不需要每次再进行搜索，从而加快训练速度
        torch.backends.cudnn.benchmark = True
        # 固定住cudnn的随机性
        torch.backends.cudnn.deterministic = True
