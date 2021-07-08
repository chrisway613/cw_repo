"""数据集划分"""

import os
import shutil
import random


def copy_img(imgs, root_dir, setname):
    """将图片拷贝至对应的目录"""

    set_dir = os.path.join(root_dir, setname)
    os.makedirs(set_dir, exist_ok=True)

    for path_img in imgs:
        print(path_img)
        shutil.copy(path_img, set_dir)

    print("[{} dataset] {} images are copied to {}".format(setname, len(imgs), set_dir))


def record_img_names(img_paths, root_dir, txt):
    """
    将数据集包含的图片文件名写入到文本文件，如训练集的图片文件名则写入到train.txt
    :param img_paths: 图片文件路径；
    :param root_dir: 数据所在的根目录
    :param txt: 记录图片文件名的文本文件
    :return:
    """

    assert txt.endswith('.txt')

    txt_path = os.path.join(root_dir, txt)
    img_names = [os.path.basename(p) for p in img_paths]

    with open(txt_path, 'w') as f:
        for name in img_names:
            f.write(name + '\n')

    set_name = txt.split('.txt')[0]
    print(f"[{set_name} dataset] image names already recorded in {txt_path}")


if __name__ == '__main__':
    # 0. config
    random_seed = 20210423
    random.seed(random_seed)

    # 按比例划分
    train_ratio = 0.8
    valid_ratio = 0.1
    # test_ratio = 0.1

    # 图片文件格式
    data_format = ".jpg"

    root_dir = '/home/cw/OpenSources/Datasets/102Flowers/data/oxford-102-flowers'
    data_dir = os.path.join(root_dir, "jpg")

    # 1. 读取图片路径并打乱
    path_imgs = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith(data_format)]
    random.shuffle(path_imgs)
    print(f"The 1st image name is: {os.path.basename(path_imgs[0])}")
    print(f"There are {len(path_imgs)} images in all")
    print('-' * 60, '\n')

    # 2. 按比例划分训练集、验证集、测试集
    train_breakpoints = int(len(path_imgs) * train_ratio)
    valid_breakpoints = int(len(path_imgs) * (train_ratio + valid_ratio))
    train_imgs = path_imgs[:train_breakpoints]
    valid_imgs = path_imgs[train_breakpoints:valid_breakpoints]
    test_imgs = path_imgs[valid_breakpoints:]

    # 3. 将对应数据集的图片拷贝到数据集对应的目录下
    copy_img(train_imgs, root_dir, "train")
    copy_img(valid_imgs, root_dir, "valid")
    copy_img(test_imgs, root_dir, "test")
    print('-' * 60, '\n')

    # 4. 将数据集下的图片文件名称记录到文本文件
    record_img_names(train_imgs, root_dir, "train.txt")
    record_img_names(valid_imgs, root_dir, "valid.txt")
    record_img_names(test_imgs, root_dir, "test.txt")
