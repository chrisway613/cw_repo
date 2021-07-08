"""将DataLoader封装起来成为迭代器，迭代器仅返回一个批次的网络输入数据和对应的标签。
   当原生的DataLoader返回的一个批次数据是map类型时，还可以继承它们自定义子类，
   重写inputs_labels_from_batch_data方法，从而获取一个批次的网络输入与标签，滤除其余无关数据信息"""

from torch.utils.data.dataloader import DataLoader


class DataLoaderIter:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self._iter = iter(dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    @staticmethod
    def inputs_labels_from_batch_data(batch_data: (list, tuple)):
        if not (isinstance(batch_data, list) or isinstance(batch_data, tuple)):
            raise ValueError(f"your batch type {type(batch_data)} is not supported,"
                             f"please inherit from `TrainDataLoaderIter` or `ValDataLoaderIter` "
                             f"and override the `inputs_labels_from_batch` method.")

        inputs, labels, *_ = batch_data
        return inputs, labels

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self._iter)
        return self.inputs_labels_from_batch_data(data)


class TrainDataLoaderIter(DataLoaderIter):
    def __init__(self, dataloader: DataLoader, auto_reset=True):
        super(TrainDataLoaderIter, self).__init__(dataloader)
        self.auto_reset = auto_reset

    def __next__(self):
        try:
            return super().__next__()
        except StopIteration:
            if not self.auto_reset:
                raise
            else:
                self._iter = iter(self.dataloader)
                return super().__next__()


class ValDataLoaderIter(DataLoaderIter):
    def __init__(self, dataloader: DataLoader):
        super(ValDataLoaderIter, self).__init__(dataloader)

        self.run_counter = 0
        self.run_limit = len(self.dataloader)

    def __iter__(self):
        if self.run_counter >= self.run_limit:
            self._iter = iter(self.dataloader)
            self.run_counter = 0

        return self

    def __next__(self):
        self.run_counter += 1
        return super().__next__()
