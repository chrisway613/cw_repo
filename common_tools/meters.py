import torch

from collections import deque, defaultdict


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
        window or the global series average."""

    def __init__(self, win_size=20):
        self.total = 0.
        self.count = 0
        self.series = []
        self.deque = deque(maxlen=win_size)

    def update(self, value):
        self.count += 1
        self.total += value

        self.deque.append(value)
        self.series.append(value)

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque)).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger:
    def __init__(self, delimeter='\t'):
        self.delimeter = delimeter
        self.meters = defaultdict(SmoothedValue)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (int, float))

            self.meters[k].update(v)

    def __str__(self):
        meters_str = [f"{name}: {meter.median:.4f}({meter.global_avg:.4f})"
                      for name, meter in self.meters.items()]
        return self.delimeter.join(meters_str)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


class AverageMeter:
    """Computes & stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt

        self.reset()

    def reset(self):
        for attr in ('val', 'avg', 'sum', 'count'):
            self.__setattr__(attr, 0)

    def update(self, val, n: int = 1):
        assert isinstance(n, int), f"'n' should be integer, current: {type(n)}"
        assert n > 0, f"n should be more than 0, got {n}"

        self.__setattr__('val', val)
        self.count += n
        self.sum += val * n
        self.__setattr__('avg', self.sum / self.count)

    def __str__(self):
        """该类对象的格式化打印输出"""
        fmt_str = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmt_str.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches: int, meters: (list, tuple), prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, print_out=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]

        # 输出到控制台
        if print_out:
            print('\t'.join(entries))

        return entries

    @ staticmethod
    def _get_batch_fmtstr(num_batches):
        # 批次数量的字符串表示占位几个字符
        num_digits = len(str(num_batches // 1))
        # 格式化
        fmt = '{:' + str(num_digits) + 'd}'

        # 用于格式化当前批次/总批次数量，比如' 1/25'(当前批次1，总共25个批次)
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    # 占位6个字符，精确到小数点后3位
    avg_meter = AverageMeter('Time', fmt=':6.3f')
    print(avg_meter)

    prog_meter = ProgressMeter(10, [avg_meter], prefix='Test: ')
    prog_meter.display(0)
