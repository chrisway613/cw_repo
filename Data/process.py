"""一些数据统计和处理，比如图像均值的统计、将不同类别的图片划分到不同的目录等"""

import os
import pickle
import shutil
import argparse
import numpy as np
import scipy.io as sio

from PIL import Image
from tqdm import tqdm


def mean_std(paths):
    """统计图像各通道均值与标准差，以便后续做归一化操作"""

    # 图像像素数量
    num = 0
    # 中间计算量
    value_sum_r = value_sum_g = value_sum_b = 0
    square_sum_r = square_sum_g = square_sum_b = 0

    for img_path in paths:
        img = Image.open(img_path).convert('RGB')
        img_array = np.asarray(img, dtype=np.float)

        h, w, _ = img_array.shape
        # 累加每张图片的像素数量
        num += h * w

        # 分别计算各通道的像素值之和和像素值平方和
        img_r = img_array[:, :, 0]
        value_sum_r += np.sum(img_r)
        square_sum_r += np.sum(np.power(img_r, 2.))

        img_g = img_array[:, :, 1]
        value_sum_g += np.sum(img_g)
        square_sum_g += np.sum(np.power(img_g, 2.))

        img_b = img_array[:, :, 2]
        value_sum_b += np.sum(img_b)
        square_sum_b += np.sum(np.power(img_b, 2.))

    # 根据公式，基于中间统计量计算各通道均值与标准差
    mean_r = value_sum_r / num
    std_r = np.sqrt(square_sum_r / num - mean_r ** 2)

    mean_g = value_sum_g / num
    std_g = np.sqrt(square_sum_g / num - mean_g ** 2)

    mean_b = value_sum_b / num
    std_b = np.sqrt(square_sum_b / num - mean_b ** 2)

    # 归一化至[0,1]
    print(mean_r / 255., std_r / 255.)
    print(mean_b / 255., std_b / 255.)
    print(mean_g / 255., std_g / 255.)


def reorder(root_dir: str):
    """
    将不同类别的图片划分到不同目录下
    :param root_dir: 数据根目录，其中包含图片目录和标签文件
    :return:
    """

    # 所有图片所在的目录
    image_dir = os.path.join(root_dir, 'jpg')
    # 标签文件路径 这里假设文件是matlab格式
    label_path = os.path.join(root_dir, 'imagelabels.mat')
    # 创建这个目录，以进一步在其中创建各个类别对应的目录
    reorder_dir = os.path.join(root_dir, 'reorder')

    image_file_names = [name for name in os.listdir(image_dir) if name.endswith('.jpg')]
    # squeeze()是因为load进来的array是(1,8189)的shape，这里压缩掉最外围的一个dim
    labels = sio.loadmat(label_path)['labels'].squeeze()
    # 图片与标签数量应该一致
    assert len(image_file_names) == len(labels)

    for file_name in tqdm(image_file_names):
        # 这里假设图像文件名称是'image_0001.jpg'这种形式，其中的序号代表类别编号
        idx = int(file_name.split('_')[1].split('.')[0])
        # 注意索引值要减1，因为图像文件名称是从1开始的；
        # 同时标签值也要减1，因为标签值也是从1开始的
        label = str(labels[idx - 1] - 1)

        # 目的目录的名称就是图像标签
        dst_dir = os.path.join(reorder_dir, label)
        # 若指定的目录不存在则创建；若存在则不影响
        os.makedirs(dst_dir, exist_ok=True)

        # 将源图像拷贝到指定目录下
        src_file_path = os.path.join(image_dir, file_name)
        shutil.copy(src_file_path, dst_dir)


def parse_err_imgs(root_dir):
    """将错误分类的图片挑出来，进行观察，图片会根据其预测情况拷贝至对应目录"""

    # pickle文件存储了错误图片信息
    parser = argparse.ArgumentParser(description='parse error images')
    parser.add_argument('--file', type=str, required=True, help='pickle file path')
    args = parser.parse_args()

    assert os.path.isfile(args.file), f'invalid file path {args.file}'
    with open(args.file, 'rb') as f:
        data = pickle.load(f)

    # 去掉后缀'.pkl'，得到目录
    out_dir = args.file[:-4]
    for set_name, info in data.items():
        for label, pred, rel_path in info:
            src_img_path = os.path.join(root_dir, os.path.basename(rel_path))
            # 目录结构：out_dir/数据集(train, valid之类的)/实际标签/预测结果
            dst_dir = os.path.join(out_dir, set_name, str(label), str(pred))
            os.makedirs(dst_dir)
            # 将源图片拷贝到目标目录下
            shutil.copy(src_img_path, dst_dir)
