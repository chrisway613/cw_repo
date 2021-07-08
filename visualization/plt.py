"""基于matplotlib.pyplot模块提供的API实现的可视化功能"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def plot_line(x_train, y_train, x_valid, y_valid, mode, out_dir):
    """
    绘制训练集和验证集的loss/acc曲线，并将图片存储到指定目录。
    :param x_train: epoch
    :param y_train: 训练集loss/acc
    :param x_valid: epoch
    :param y_valid: 验证集loss/acc
    :param mode: 'Loss' or 'Acc'
    :param out_dir: 输出路径
    :return:
    """

    plt.plot(x_train, y_train, label='Train')
    plt.plot(x_valid, y_valid, label='Valid')

    plt.ylabel(mode)
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'Loss' else 'upper left'
    plt.legend(loc=location)
    plt.title('_'.join([mode]))

    # 若输出目录不存在，则创建。注意，若中间有任何目录不存在，都会创建
    # 设置exist_ok=True，则目录存在的情况下不会抛出异常。
    os.makedirs(out_dir, exist_ok=True)
    out_name = mode + '.png'
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path)

    plt.close()


def plot_bar(root_dir: str, out_path: str, display=False):
    """
    画出数据分布(比如类别分布)的条形图，并存储至图片文件。
    :param root_dir: 数据所在的根目录，在该目录下需要有图片文件及标签文件；
    :param out_path: 可视化结果输出的目标路径；
    :param display: 是否即时显示
    :return:
    """

    # 图片目录
    data_dir = os.path.join(root_dir, 'images')
    # 标签文件
    label_path = os.path.join(root_dir, 'imagelabels.mat')
    # 这里假设标签文件是.mat格式，即matlab文件
    # 使用scipy.io.loadmat()方法可以加载此文件，该方法返回一个dict
    # key->np.ndarray (1,num_data)
    labels = sio.loadmat(label_path)['labels'].squeeze()
    # 类别数量
    cls_num = np.max(labels)
    # 每类样本数量：这个list的索引对应每一类，值对应每类的样本数量
    num_per_cls = [0] * cls_num

    # 每张图片的文件名
    for name in os.listdir(data_dir):
        # 这里假设图片文件是'image_0001.jpg'这种，
        # 其中的序号就代表其类别编号，但是要转换成索引就得从0开始
        # 于是要减1
        idx = int(name.split('_')[1].split('.')[0]) - 1
        # 标签中的值也是从1开始的，转换成索引也需要减1
        label = labels[idx] - 1
        # 数量统计
        num_per_cls[label] += 1

    x = np.arange(cls_num)
    # 画出类别分布的条形图
    plt.bar(x, num_per_cls, width=.6, color='b', align='center')

    # 设置坐标轴的标记
    plt.xlabel('classes')
    plt.ylabel('sample counts')

    plt.xticks(x)
    # plt.yticks(num_per_cls)
    # 设置标题
    plt.title('Sample Counts of ALl Categories')

    # 保存文件
    plt.savefig(out_path)
    if display:
        plt.show()


def show_conf_mat(conf_mat: np.ndarray, classes: (list, tuple), set_name: str, out_dir: str,
                  epoch: int, verbose=False, perc=False):
    """
    绘制混淆矩阵，并且输出到指定目录。
    :param conf_mat: 混淆矩阵；
    :param classes: 类别名称；
    :param set_name: 数据集：'train', 'valid' or 'test'；
    :param out_dir: 输出目录；
    :param epoch: 当前是第几个周期；
    :param verbose: 是否打印日志；
    :param perc: 是否采用百分比形式，通常在图像分割任务中使用，因分类数目过大
    :return:
    """

    # 类别数量
    cls_num = len(classes)

    # 归一化
    conf_mat_normalized = conf_mat.copy()
    for i in range(cls_num):
        conf_mat_normalized[i, :] = conf_mat_normalized[i, :] / conf_mat_normalized[i].sum()

    # 设置画布大小(figure size)
    if cls_num < 10:
        fig_size = 6
    elif cls_num >= 100:
        fig_size = 30
    else:
        # 区间是[6,30]，总共91个点，均匀分布
        fig_size = np.linspace(6, 30, 91)[cls_num - 10]
    # 设置画布尺寸
    plt.figure(figsize=(int(fig_size), int(fig_size * 1.3)))

    # 设置颜色
    # 更多颜色请参考: http://matplotlib.org/examples/color/colormaps_reference.html
    color_map = plt.get_cmap('Greys')
    # 绘制混淆矩阵(绘制在画布上，并不会弹窗显示，除非使用plt.show())
    plt.imshow(conf_mat_normalized, cmap=color_map)
    # 颜色条，由浅至深指示了数据的不同程度
    plt.colorbar(fraction=.03)

    # 设置标记：坐标轴、标题
    plt.ylabel('True')
    plt.xlabel('Predict')
    plt.title(f'Confusion_Matrix_{set_name}_{epoch}')

    loc = list(range(cls_num))
    plt.yticks(loc, list(classes))
    plt.xticks(loc, list(classes), rotation=60)

    # 设置文字
    # 百分比形式
    if perc:
        # 各类别的TP+FP
        pred_num_per_cls = conf_mat.sum(axis=0)
        # 对角线上是TP/(TP+FP)即precision，其它位置是FP/(TP+FP)
        conf_mat_perc = conf_mat / pred_num_per_cls
        for i in range(cls_num):
            for j in range(cls_num):
                plt.text(x=j, y=i, s='{:.0%}'.format(conf_mat_perc[i, j]),
                         va='center', ha='center', color='red', fontsize=10)
    # 原始形式，也就是每个值都是数量
    else:
        for i in range(cls_num):
            for j in range(cls_num):
                plt.text(x=j, y=i, s=int(conf_mat[i, j]),
                         va='center', ha='center', color='red', fontsize=10)

    # 将绘制混淆矩阵的图片输出到指定目录
    file_name = f'Confusion_Matrix_{set_name}.png'
    # 若输出目录不存在则创建(存在也没关系)
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, file_name)
    plt.savefig(file_path)

    plt.close()

    # 打印日志(通常在最后一个epoch时)
    if verbose:
        # 用于避免分母为0导致除法结果溢出的极小值
        eps = 1e-9
        for idx in range(cls_num):
            print("set:{}, class:{:<10}, total num:{:<6}, correct num:{:<5} "
                  "Recall:{:.2%}, Precision:{:.2%}\n".format(set_name, classes[idx],
                                                             conf_mat[idx].sum(), conf_mat[idx, idx],
                                                             conf_mat[idx, idx] / (conf_mat[idx].sum() + eps),
                                                             conf_mat[idx, idx] / (conf_mat[:, idx].sum() + eps))
                  )
