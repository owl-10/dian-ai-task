import numpy as np
import torch


def compute_iou(bbox1, bbox2):

    # TODO Compute IoU of 2 bboxes.
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    inter_x1 = np.maximum(bbox1[:, 0], bbox2[:, 0])
    inter_x2 = np.minimum(bbox1[:, 2], bbox2[:, 2])
    inter_y1 = np.maximum(bbox1[:, 1], bbox2[:, 1])
    inter_y2 = np.minimum(bbox1[:, 3], bbox2[:, 3])

    intersaction = np.maximum(inter_x2-inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    union = area1 + area2 - intersaction
    iou = intersaction / union
    return iou
    ...

    # End of todo
