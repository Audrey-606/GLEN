import sys
sys.path.append("/hy-tmp/BlendMask/detectron2")
sys.path.append("/hy-tmp/BlendMask/AdelaiDet")
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from adet.config import get_cfg as get_adet_cfg
from typing import List, Dict

class FeatureHook:
    def __init__(self):
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output.detach().cpu()


class FeatureVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.checkpointer = DetectionCheckpointer(self.model)
        self.hooks = {}
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.size_divisibility = self.model.backbone.size_divisibility

    def _find_layer(self, name):
        """支持ModuleDict的层查找方法"""
        module = self.model
        for part in name.split("."):
            if isinstance(module, nn.ModuleDict):
                module = module[part]
            else:
                module = getattr(module, part)
        return module

    def register_hook(self, layer_path):
        target_layer = self._find_layer(layer_path)
        hook = FeatureHook()
        handle = target_layer.register_forward_hook(hook.hook_fn)
        self.hooks[layer_path] = (hook, handle)
        return self

    def remove_hooks(self):
        for _, handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()

    def preprocess_image(self, image_path):
        """与BlendMask完全一致的预处理流程"""
        original_image = read_image(image_path, format="BGR")
        height, width = original_image.shape[:2]

        image = original_image.astype(np.float32)  # 确保使用float32
        image = image[:, :, ::-1]  # BGR转RGB

        mean = np.array(self.cfg.MODEL.PIXEL_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.cfg.MODEL.PIXEL_STD, dtype=np.float32).reshape(1, 1, 3)
        image = (image - mean) / std

        image_tensor = torch.as_tensor(
            image.transpose(2, 0, 1),
            dtype=torch.float32  # 新增类型指定
        )

        image_list = ImageList.from_tensors(
            [image_tensor.to(self.device)],
            self.size_divisibility
        )
        return original_image, image_list

    def visualize_features(self, features, output_dir, layer_name,original_image, prefix="",
                          heatmap_style=False, channel_limit=64):
        """改进的特征可视化方法"""
        os.makedirs(output_dir, exist_ok=True)

        if features.dim() == 4:  # [batch, channel, h, w]
            features = features[0]
        elif features.dim() == 3:  # [channel, h, w]
            pass
        else:
            raise ValueError(f"不支持的维度: {features.dim()}")

        # 生成热力图模式
        if heatmap_style:
            # 改进的热力图生成逻辑
            with torch.no_grad():
                # 全局归一化
                feat_np = features.cpu().numpy()
                min_val = feat_np.min()
                max_val = feat_np.max()
                feat_norm = (feat_np - min_val) / (max_val - min_val + 1e-8)

                # 多通道融合策略
                channel_weights = np.abs(feat_norm).mean(axis=(1, 2))  # 使用绝对值均值加权
                cam = np.sum(feat_norm * channel_weights[:, np.newaxis, np.newaxis], axis=0)

                # 增强对比度
                cam = cv2.normalize(cam, None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cam = (cam * 255).astype(np.uint8)

                # 应用热力图配色
                canvas = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

                # 调整原始图像尺寸与热力图匹配
                h, w = original_image.shape[:2]
                canvas = cv2.resize(canvas, (w, h))  # 调整热力图尺寸

                # 转换原始图像颜色空间
                original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                # 叠加显示
                overlay = cv2.addWeighted(original_rgb, 0.5, canvas, 0.5, 0)
                canvas = overlay
        else:
            # 通道网格模式
            num_channels = min(channel_limit, features.shape[0])
            grid_size = int(np.ceil(np.sqrt(num_channels)))
            feature_h, feature_w = features.shape[1], features.shape[2]
            canvas = np.zeros((grid_size * feature_h, grid_size * feature_w), dtype=np.float32)

            for i in range(num_channels):
                row = i // grid_size
                col = i % grid_size
                channel_feat = features[i].numpy()
                channel_feat = (channel_feat - channel_feat.min()) / (channel_feat.max() - channel_feat.min() + 1e-8)
                canvas[row * feature_h:(row + 1) * feature_h,
                       col * feature_w:(col + 1) * feature_w] = channel_feat

            canvas = (canvas * 255).astype(np.uint8)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        output_path = os.path.join(output_dir, f"{prefix}{layer_name.replace('.', '_')}.jpg")
        cv2.imwrite(output_path, canvas)
        return output_path

    def process_image(self, image_path, output_dir, layer_name, heatmap_style=False):
        original_img, image_list = self.preprocess_image(image_path)

        inputs = [{"image": image_list.tensor[0]}]

        with torch.no_grad():
            self.model(inputs)

        features = self.hooks[layer_name][0].features
        if features is None:
            raise RuntimeError(f"未捕获到{layer_name}层的特征")

        return self.visualize_features(
            features, output_dir, layer_name,
            original_image=original_img,
            prefix=os.path.basename(image_path).split('.')[0] + "_",
            heatmap_style=heatmap_style
        )


def setup_cfg(args):
    cfg = get_adet_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--featuremap-layer", required=True)
    parser.add_argument("--heatmap-style", action="store_true")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_logger()
    cfg = setup_cfg(args)

    visualizer = FeatureVisualizer(cfg)
    visualizer.register_hook(args.featuremap_layer)
    visualizer.checkpointer.load(cfg.MODEL.WEIGHTS)

    if os.path.isfile(args.input):
        output_path = visualizer.process_image(
            args.input, args.output_dir, args.featuremap_layer,
            heatmap_style=args.heatmap_style
        )
        print(f"特征图已保存至: {output_path}")
    else:
        raise ValueError("输入路径必须是单个图片文件")

    visualizer.remove_hooks()


if __name__ == "__main__":
    main()