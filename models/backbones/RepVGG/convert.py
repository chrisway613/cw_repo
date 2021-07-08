"""Convert the training-time models into inference-time"""

import os
import torch
import argparse

from .repvgg import get_RepVGG_func_by_name, repvgg_model_convert


def parse_args():
    parser = argparse.ArgumentParser(description='RepVGG Conversion')
    # 训练好的权重路径
    parser.add_argument('load', metavar='LOAD', help='path to the training time weights file')
    # 转换成推理时结构后的权重保存到的目的路径
    parser.add_argument('save', metavar='SAVE', help='path to the inference-time weights file')
    # 哪一种模型
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_build_func = get_RepVGG_func_by_name(args.arch)
    train_model = model_build_func(deploy=False)

    if os.path.isfile(args.load):
        print(f"=> loading checkpoint '{args.load}'")

        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        # strip the names
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        train_model.load_state_dict(ckpt)
    else:
        print(f"=> no checkpoint found at '{args.load}'")

    repvgg_model_convert(train_model, save_path=args.save)
    print(f"=> converted weights saved at '{args.save}'\n")
