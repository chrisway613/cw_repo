"""计算模型推断的帧率，即fps(frame per seconds)"""

import time
import torch

from torchvision.models import vgg16
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    ds = TensorDataset(torch.rand(4, 3))
    test_loader = DataLoader(ds, batch_size=2, num_workers=4, pin_memory=True)

    model = vgg16(pretrained=True)
    model.eval()

    inf_time = 0.
    num_frames = 0
    time_warmup = 5
    log_interval = 10

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, path_imgs = data
            if not labels.dtype == torch.long:
                labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)

            # 计算模型推断耗时
            # 正确测试推断时间，
            # 一定要加上这行代码，该操作是等待GPU全部执行结束，CPU才可以读取时间信息。
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(inputs)
            # 再次同步，等待当前设备上所有流中的所有核心完成
            # 可以指定device，若没有指定，则是当前使用的device
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            # 前几个周期可能会比较慢，因此忽略
            if i < time_warmup:
                continue

            inf_time += elapsed
            # 由于最后一个batch可能不足batch size
            # 因此累加计数的时候应该加上当前batch的数据量
            num_frames += len(labels)

            if (i + 1) % log_interval == 0:
                # 当前的fps
                fps = len(labels) / elapsed
                # 全局平均fps
                global_fps = num_frames / inf_time
                print(f"Done batch[{i + 1}/{len(test_loader)}] fps={fps:.1f}(Avg {fps:.1f})")

    global_fps = num_frames / inf_time
    print(f"Overall fps={global_fps:.1f}")
