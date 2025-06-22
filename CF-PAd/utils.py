import os
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_curve, auc


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @staticmethod
    def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
        """
        Computes the precision@k for the specified values of k
        """
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def get_eer_threshold_cross_db(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray):
    """
    Calculate Equal Error Rate (EER) and threshold where FPR and FNR are closest.
    """
    differ_tpr_fpr = tpr + fpr - 1.0
    right_index = np.nanargmin(np.abs(differ_tpr_fpr))
    best_th = thresholds[right_index]
    eer = fpr[right_index]
    return eer, best_th, right_index


def performances_cross_db(prediction_scores: np.ndarray, gt_labels: np.ndarray, pos_label=1, verbose=True):
    """
    Evaluate model performance with ROC, AUC, EER, HTER, APCER, BPCER.
    """
    fpr, tpr, thresholds = roc_curve(gt_labels, prediction_scores, pos_label=pos_label)
    val_eer, val_threshold, right_index = get_eer_threshold_cross_db(fpr, tpr, thresholds)
    test_auc = auc(fpr, tpr)

    FRR = 1 - tpr  # False Reject Rate
    HTER = (fpr + FRR) / 2.0  # Half Total Error Rate

    if verbose:
        print(f'AUC@ROC: {test_auc:.4f}, HTER: {HTER[right_index]:.4f}, APCER: {fpr[right_index]:.4f}, '
              f'BPCER: {FRR[right_index]:.4f}, EER: {val_eer:.4f}, Threshold: {val_threshold:.4f}')

    return test_auc, fpr[right_index], FRR[right_index], HTER[right_index]


def evaluate_threshold_based(prediction_scores: list, gt_labels: list, threshold: float):
    """
    Calculate APCER, BPCER, ACER at a given threshold.
    """
    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    num_real = sum(s['label'] == 1 for s in data)
    num_fake = sum(s['label'] == 0 for s in data)

    type1 = sum(s['map_score'] <= threshold and s['label'] == 1 for s in data)  # False Rejects
    type2 = sum(s['map_score'] > threshold and s['label'] == 0 for s in data)   # False Accepts

    test_threshold_APCER = type2 / num_fake if num_fake else 0
    test_threshold_BPCER = type1 / num_real if num_real else 0
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return test_threshold_APCER, test_threshold_BPCER, test_threshold_ACER


def compute_video_score(video_ids: list, predictions: list, labels: list):
    """
    Aggregates frame-level predictions and labels to video-level by averaging.
    """
    predictions_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    for vid, pred, label in zip(video_ids, predictions, labels):
        predictions_dict[vid].append(pred)
        labels_dict[vid].append(label)

    new_predictions, new_labels, new_video_ids = [], [], []

    for vid in set(video_ids):
        avg_score = np.mean(predictions_dict[vid])
        label = labels_dict[vid][0]  # assuming all frames share the same label
        new_video_ids.append(vid)
        new_predictions.append(avg_score)
        new_labels.append(label)

    return new_predictions, new_labels, new_video_ids
