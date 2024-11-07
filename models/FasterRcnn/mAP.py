import torch
import numpy as np
import matplotlib.pyplot as plt


def iou(rec1, rec2):
    x1, y1, x2, y2 = rec1
    x3, y3, x4, y4 = rec2
    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    overlap = x_overlap * y_overlap
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - overlap
    return overlap / union


def get_tps(model, dataloader, num_classes, device, score_threshold=0, iou_threshold=0.5):
    model.eval()
    model.to(device)
    model.roi_heads.score_thresh = score_threshold
    tps = [[] for i in range(num_classes)]  # [class1, class2, ...]
    scores = [[] for i in range(num_classes)]  # [class1, class2, ...]
    n_gts = [0 for i in range(num_classes)]  # [class1, class2, ...]
    for image, target in dataloader:
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        with torch.no_grad():
            prediction = model(image)[0]
        gt_num = target['boxes'].squeeze(0).shape[0]
        pred_num = prediction['boxes'].shape[0]
        gt = np.hstack([target['boxes'].squeeze(0).cpu().numpy(), target['labels'].squeeze(0).unsqueeze(-1).cpu().numpy(), np.zeros([gt_num, 1])])  # [x1, y1, x2, y2, label, 0]
        pred = np.hstack([prediction['boxes'].cpu().numpy(), prediction['labels'].unsqueeze(-1).cpu().numpy(), prediction['scores'].unsqueeze(-1).cpu().numpy(), np.zeros([pred_num, 1])])  # [x1, y1, x2, y2, label, score, 0]
        gts = [gt[gt[:, 4] == i] for i in range(1, num_classes+1)]
        preds = [pred[pred[:, 4] == i] for i in range(1, num_classes+1)]

        for i in range(num_classes):
            preds[i] = preds[i][preds[i][:, 5].argsort()][::-1]  # 按score从大到小排序
            for j in range(preds[i].shape[0]):
                iou_pre = 0
                temp_k = 0
                for k in range(gts[i].shape[0]):
                    if gts[i][k, -1] != 1:
                        iou_temp = iou(preds[i][j, :4], gts[i][k, :4])
                        if iou_temp > iou_threshold and iou_temp > iou_pre:
                            iou_pre = iou_temp
                            temp_k = k
                if iou_pre != 0:
                    preds[i][j, -1] = 1
                    gts[i][temp_k, -1] = 1
                    tps[i].append(1)
                    scores[i].append(preds[i][j, -2])
                else:
                    tps[i].append(0)
                    scores[i].append(preds[i][j, -2])
            n_gts[i] += gts[i].shape[0]
    return tps, scores, n_gts


def get_mAP(tps, scores, n_gts, num_classes):
    ap = 0
    recalls, precisions = [], []
    for i in range(num_classes):
        recall, precision = get_PR(tps[i], scores[i], n_gts[i])
        recalls.append(recall)
        precisions.append(precision)
        ap += get_AP(recall, precision)
    return ap/num_classes, recalls, precisions


def get_PR(tp, score, n_gt):
    tp = np.array(tp)
    tp = tp[np.argsort(score)[::-1]]
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / (np.arange(tp.shape[0]) + 1)
    return recall, precision


def get_AP(recall, precision):
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    for i in range(precision.shape[0] - 2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices-1]) * precision[indices])
    return ap


def calculate_mAP(model, dataloader, num_classes, device, score_threshold=0, iou_threshold=0.5):
    tps, scores, n_gts = get_tps(model, dataloader, num_classes, device, score_threshold, iou_threshold)
    return get_mAP(tps, scores, n_gts, num_classes)


def draw_PR(recall, precision, name):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='o', linestyle='-', color='b')

    # 为每个点添加文本标签
    # for i in range(len(precision)):
    #     plt.text(recall[i], precision[i], f'({recall[i]:.2f}, {precision[i]:.2f})')

    # 设置图表的标题和坐标轴标签
    plt.title(name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # 显示网格（可选）
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)

    # 显示图例（可选）
    plt.legend(['PR Curve'])

    # 显示图表
    plt.show()


if __name__ == "__main__":
    gt_labels = np.load("F:\\QBC\\STEM-Pytorch\\mAP\\gt\\labels.npy")
    gt_boxes = np.load("F:\\QBC\\STEM-Pytorch\\mAP\\gt\\boxes.npy")
    pred_labels = np.load("F:\\QBC\\STEM-Pytorch\\mAP\\pred\\labels.npy")
    pred_boxes = np.load("F:\\QBC\\STEM-Pytorch\\mAP\\pred\\boxes.npy")
    pred_scores = np.load("F:\\QBC\\STEM-Pytorch\\mAP\\pred\\scores.npy")
    gt_num = gt_labels.shape[0]
    pred_num = pred_labels.shape[0]

    gt = np.hstack([gt_boxes, gt_labels[:, np.newaxis], np.zeros([gt_num, 1])])  # [x1, y1, x2, y2, label, 0]
    pred = np.hstack([pred_boxes, pred_labels[:, np.newaxis], pred_scores[:, np.newaxis], np.zeros([pred_num, 1])])  # [x1, y1, x2, y2, label, score, 0]
    gts = [gt[gt[:, 4] == i] for i in range(1, 4)]
    preds = [pred[pred[:, 4] == i] for i in range(1, 4)]
    tps = [[] for i in range(3)]
    scores = [[] for i in range(3)]
    n_gts = [0 for i in range(3)]

    for i in range(3):
        preds[i] = preds[i][preds[i][:, 5].argsort()][::-1]
        for j in range(preds[i].shape[0]):
            iou_pre = 0
            temp_j = 0
            temp_k = 0
            for k in range(gts[i].shape[0]):
                if gts[i][k, -1] != 1:
                    iou_temp = iou(preds[i][j, :4], gts[i][k, :4])
                    if iou_temp > 0.5 and iou_temp > iou_pre:
                        iou_pre = iou_temp
                        temp_j = j
                        temp_k = k
            if iou_pre != 0:
                preds[i][temp_j, -1] = 1
                gts[i][temp_k, -1] = 1
                tps[i].append(1)
                scores[i].append(preds[i][j, -2])
            else:
                tps[i].append(0)
                scores[i].append(preds[i][j, -2])
        n_gts[i] += gts[i].shape[0]
    print(get_mAP(tps, scores, n_gts, 3))
    # for i in range(3):
    #     recall, precision = get_PR(tps[i], scores[i], n_gts[i])
    #     draw_PR(recall, precision, f"Class {i+1}")
