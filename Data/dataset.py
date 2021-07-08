"""数据集实例的构建，通常实现一个继承torch.utils.data.dataset.Dataset的子类，
需要自定义的方法包括__getitem__(), __len__()"""

import os
import torch
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class MyDataset(Dataset):
    cls_num = 102

    # 图片数据格式
    data_format = '.jpg'
    # 数据集类型
    datasets = ('train', 'valid', 'test')

    # 标签文件，这里假设是matlab格式，使用scipy.io.loadmat()方法可以读取
    label_file = 'imagelabels.mat'

    def __init__(self, root_dir, transform=None):
        """

        :param root_dir: 数据集目录，该目录下有图片文件
        :param transform: 若使用数据增强，则它是torchvision.transforms模块中的某些方法
        """

        self.root_dir = root_dir
        assert os.path.isdir(root_dir), f"got invalid directory: {root_dir}"

        # 数据集名称，'train'/'valid'/'test'，这里假设数据集的最后一级目录即其名称
        set_name = os.path.basename(root_dir)
        assert set_name in self.datasets, f"dataset must be 'train', 'valid' or 'test', got {set_name}"

        self.transform = transform

        # 数据集所有图片的相关信息，方便回溯
        # [(path, label), ... , ]
        self.img_info = []
        # 数据集所有样本的标签
        self.label_array = None

        # 调用该方法会设置self.img_info和self.label_array
        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，然后预处理，最后转换成张量(to tensor)
            :param index: 数据索引
            :return: 图像、对应的标签
        """

        path_img, label = self.img_info[index]
        # 为了便于处理，最好转成RGB格式，避免后续处理时在通道维度出错
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        # 该部分可视项目实际情况处理
        else:
            # 需要注意的是要将通道维度置于最前面
            # PIL Image->array->tensor & (H,W,C)->(C,H,W)
            img = torch.from_numpy(np.asarray(img, dtype=np.float)).permute((2, 0, 1)).contiguous()

        return img, label, path_img

    def __len__(self):
        """

        :return: 数据集样本数量
        """

        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取，将硬盘中的数据路径、标签读取进来，存在list中，
        其中每项是(path, label)。
        :return:
        """

        # 数据集图像路径
        # img_paths = list(map(lambda n: os.path.join(self.root_dir, n), os.listdir(self.root_dir)))
        # 使用列表生成式较以上使用方式效率更高
        img_paths = [os.path.join(self.root_dir, n) for n in os.listdir(self.root_dir)]
        # 图片文件夹的上一级目录，这里假设那个目录中存有标签文件
        prefix_dir = os.path.dirname(self.root_dir)
        # 标签文件路径
        label_path = os.path.join(prefix_dir, self.label_file)

        # sio.loadmat(label_path)返回dict，key 'labels'对应的是numpy.ndarray，里面是各图像对应的标签值(从1开始)
        # shape是(1,n_data)，因此squeeze()压缩最外层的一维
        labels = sio.loadmat(label_path)['labels'].squeeze()
        # 标签最大值不超过类别数量(102)
        assert labels.max() <= self.cls_num
        # 标签是所有图像(各个数据集)的标签，因此数量上应满足以下关系
        assert len(labels) >= len(img_paths)

        # 图片文件名称中的序号(从1开始)
        # img_indices = list(
        #     map(
        #         lambda n: int(n.split('_')[1].split(self.data_format)[0]),
        #         os.listdir(self.root_dir)
        #     )
        # )
        # 使用列表生成式效率较以上使用方式高
        # 这里假设图片文件名称都是'image_0001.jpg'这种形式，其中的序号代表其类别编号(如0001代表类别1)
        img_indices = [int(n.split('_')[1].split(self.data_format)[0]) for n in os.listdir(self.root_dir)]
        # 序号最大值不应超过图片数量
        assert max(img_indices) <= len(labels)
        # 获取图片对应的标签值，注意索引需要是序号值减1，并且标签值也要减1(这里假设不需要背景类)，使其从0开始
        # 注意加上int()，这样后续就不需要在训练pipeline中将label值转换为long(通常使用CrossEntropyLoss等都需要标签是long类型)
        self.label_array = np.asarray([int(labels[index - 1] - 1) for index in img_indices])
        self.img_info = list(zip(img_paths, self.label_array))


if __name__ == '__main__':
    # 缩放分辨率
    S = 256
    # 裁减区域大小
    crop_area = 224

    # ImageNet图像各通道均值
    imagenet_mean = [.485, .456, .406]
    # ImageNet图像各通道标准差
    imagenet_std = [.229, 0.224, .225]

    # 数据集根目录
    root_dir = '/home/cw/OpenSources/Datasets/102Flowers/data/oxford-102-flowers/train'

    transform = transforms.Compose([
        # 最短边缩放到S，另一边按等比例缩放
        # 注意和(S,S)的区别，后者是缩放成方形
        transforms.Resize(S),
        # 中心裁剪出(S,S)大小
        transforms.CenterCrop(S),
        # 随机裁剪
        transforms.RandomCrop(crop_area),
        # 随机水平翻转，默认翻转概率为0.5
        transforms.RandomHorizontalFlip(),
        # 颜色扰动
        transforms.ColorJitter(brightness=.2, contrast=.1, saturation=.2, hue=.1),
        # 转换成[0,1]张量
        transforms.ToTensor(),
        # 标准化(0均值，1方差)
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    train_dataset = MyDataset(root_dir, transform=transform)
    dataset_iter = iter(train_dataset)
    _, __ = next(dataset_iter)
    img_ts, label = next(dataset_iter)

    print(f"Total {len(train_dataset)} images in dataset\n")
    print(f"The 2nd image info is:\n {train_dataset.img_info[1]}\n")
    print(f"tensor:\n {img_ts}\nshape: {img_ts.shape}\ndata type: {img_ts.dtype}\n")
    print(f"label: {label}\n")

    bs = 16
    workers = 4
    # 对于训练集：
    # shuffle=True 打乱数据顺序，避免过拟合；
    # drop_last=True 当最后一个batch的数据量不足batch size时就舍弃掉，避免其样本数量太少造成模型学到的是个例而非整体分布
    # num_workers通常搭配pin_memory使用，前者开启多线程加载数据(通常最大可设置为CPU核心数量)；
    # 而pin_memory和内存锁页相关，通常也能加快数据加载流程。
    loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    for i, data in enumerate(loader):
        print(f"batch {i}:")
        imgs, labels, img_paths = data
        print(f"images:\n{imgs}\n")
        print(f"labels:\n{labels}\n")
        print(f"image paths:\n{img_paths}\n")
