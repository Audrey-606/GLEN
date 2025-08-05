import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.ndimage import sobel, binary_erosion


class DualTaskMetrics:
    def __init__(self, max_length=300):  # 假设最大允许长度为300mm
        self.seg_metrics = {
            'boundary_f1': [],
            'pred_masks': [],
            'gt_masks': []
        }
        self.length_metrics = {
            'pred_lengths': [],
            'gt_lengths': []
        }
        self.max_length = max_length

    def update_boundary_f1(self, pred_mask, gt_mask):
        """
        计算单个实例的边界F1分数
        """

        def get_edge(mask):
            erosion = binary_erosion(mask)
            return mask ^ erosion

        pred_edge = get_edge(pred_mask)
        gt_edge = get_edge(gt_mask)

        overlap = np.logical_and(pred_edge, gt_edge).sum()
        precision = overlap / (pred_edge.sum() + 1e-6)
        recall = overlap / (gt_edge.sum() + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return f1

    def update_seg(self, outputs, targets):
        """
        累积分割指标数据
        """
        for output, target in zip(outputs, targets):
            pred_masks = output['instances'].pred_masks.cpu().numpy()
            gt_masks = target['instances'].gt_masks.cpu().numpy()

            for pred, gt in zip(pred_masks, gt_masks):
                f1 = self.update_boundary_f1(pred, gt)
                self.seg_metrics['boundary_f1'].append(f1)

    def update_length(self, outputs, targets):
        """
        累积长度指标数据
        """
        for output, target in zip(outputs, targets):
            preds = output['instances'].pred_lengths.cpu().numpy()
            gts = target['instances'].gt_pixel_length.cpu().numpy()

            self.length_metrics['pred_lengths'].extend(preds)
            self.length_metrics['gt_lengths'].extend(gts)

    def compute_seg_metrics(self, coco_eval):
        """
        计算分割相关指标
        """
        # 从COCO评估中获取mAP
        stats = coco_eval.stats
        seg_metrics = {
            'mAP_50': stats[1],  # COCO mAP@0.5
            'mAP_75': stats[2],  # COCO mAP@0.75
            'boundary_f1': np.nanmean(self.seg_metrics['boundary_f1'])
        }
        return seg_metrics

    def compute_length_metrics(self):
        """
        计算长度预测指标
        """
        pred = np.array(self.length_metrics['pred_lengths'])
        gt = np.array(self.length_metrics['gt_lengths'])

        # 过滤无效值
        valid_idx = ~np.isnan(pred) & ~np.isnan(gt) & (gt > 0)
        pred = pred[valid_idx]
        gt = gt[valid_idx]

        return {
            'MAE': mean_absolute_error(gt, pred),
            'RMSE': np.sqrt(mean_squared_error(gt, pred)),
            'R2': r2_score(gt, pred)
        }

    def composite_score(self, seg_metrics, length_metrics):
        """
        计算复合指标
        """
        mAP = seg_metrics['mAP_50']
        mae = length_metrics['MAE']

        # 归一化MAE (假设最大允许误差为50mm)
        mae_norm = 1 - min(mae / 50, 1.0)
        return 0.7 * mAP + 0.3 * mae_norm