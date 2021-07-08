"""模型相关评估指标"""

import torch

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve, roc_auc_score


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        max_k = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
        # (b,max_k)
        correct = pred.eq(target.unsqueeze(-1).expand_as(pred))

        result = []
        for k in topk:
            # (1,)
            correct_k = correct[:, :k].reshape(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100. / batch_size))

        # a list of length k
        return result


def roc_auc_sklearn(labels, scores, pos_label=None, display=False):
    """利用sklearn的API来绘制ROC曲线和计算AUC值
       当标签值不是非0即1或者非-1即1时，需要指定pos_laebl，即正例的标签值。"""

    # 自适应阀值，分别将各个预测结果的值作为阀值来绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
    # 利用各种阀值下统计得到的FPR与TPR计算AUC
    auc_value = auc(fpr, tpr)

    # 输出计算结果以及绘制曲线
    if display:
        print(f"FPR:{fpr}")
        print(f"TPR:{tpr}")
        print(f"threshold:{thresholds}")
        print(f"auc: {auc_value}\n")

        plt.plot(fpr, tpr, 'b--')
        plt.title('ROC')

        for x, y in zip(fpr, tpr):
            plt.text(x, y, (x, y), fontsize=8)

        plt.show()

    return fpr, tpr, thresholds, auc_value


def roc_auc(labels, scores, pos_label=None, display=False):
    """绘制ROC曲线和计算AUC值"""
    
    # i. 统计FPR和TPR，以绘制ROC曲线

    # 若没有显式指定正类的标签值，则默认为1
    if pos_label is None:
        pos_label = 1

    # 正负样本数量
    num_pos = (labels == pos_label).sum()
    num_neg = len(labels) - num_pos

    # TP/(TP+FN)
    tpr = []
    # FP/(FP+TN)
    fpr = []
    # 自适应阀值，将预测值分别作为各个阀值
    # 注意要去重！
    thresholds = [1 + max(scores)] + sorted(np.unique(scores), reverse=True)

    # 基于各个阀值统计FPR和TPR
    for t in thresholds:
        tp = ((scores >= t) & (labels == 1)).sum()
        fp = ((scores >= t) & (labels == 0)).sum()

        tpr.append(tp / num_pos if num_pos else 0.)
        fpr.append(fp / num_neg if num_neg else 0.)

    # ii. 计算AUC(使用物理意义计算，即：任意抽取一对正负样本，对正样本打分高于负样本的概率)

    # 先对预测值进行排序
    sorted_scores = np.sort(scores)
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    rank = np.arange(1, len(labels) + 1)
    # 对于score相等的样本的rank，取均值
    for s, c in zip(*np.unique(scores, return_counts=True)):
        if c > 1:
            indices = sorted_scores == s
            rank[indices] = rank[indices].mean()
    # print(f"rank: {rank}\n")

    pos_indices = sorted_labels == pos_label
    auc = (rank[pos_indices].sum() - num_pos * (1 + num_pos) / 2) / (num_pos * num_neg)

    # 输出统计结果和绘制ROC曲线
    if display:
        print(f"FPR:{fpr}")
        print(f"TPR:{tpr}")
        print(f"threshold:{thresholds}")
        print(f"auc: {auc}")

        plt.plot(fpr, tpr, 'r--')
        plt.title('ROC')

        for x, y in zip(fpr, tpr):
            plt.text(x, y, (x, y), fontsize=8)

        plt.show()

    return fpr, tpr, thresholds, auc


if __name__ == '__main__':
    np.random.seed(0)

    # labels = np.random.randint(0, 2, 10)
    # scores = np.array([0.5, 0.5, 0.3, 0.2, 0.1, 0.6, 0.5, 0.6, 0.8, 0.6])
    # print(f"labels:\n{labels}")
    # print(f"scores:\n{scores}")
    #
    # print(f"compute by API:\n")
    # fpr, tpr, threshold, auc = roc_auc_sklearn(labels, scores, display=True)
    # print(f"comput by self-defined:\n")
    # fpr_, tpr_, threshold_, auc_ = roc_auc(labels, scores, display=True)
    #
    # auc_value = roc_auc_score(labels, scores)
    # print(f"\nauc: {auc_value}")
    #
    # assert np.all([
    #     (fpr == fpr_).all(),
    #     (tpr == tpr_).all(),
    #     (threshold == threshold_).all(),
    #     auc == auc_ == auc_value
    # ])

    outputs = torch.randn(2, 10)
    predictions = torch.softmax(outputs, -1)
    targets = torch.randint(10, (2,))
    print(f"predictions:\n{predictions}")
    print(f"targets:\n{targets}")
    acc1, acc5 = accuracy(predictions, targets, topk=(1, 5))
    print(f"Top1-Acc: {acc1.item()}, Top5-Acc: {acc5.item()}")
