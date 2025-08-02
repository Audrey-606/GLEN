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
from tqdm import tqdm
import matplotlib.cm as cm


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

        # 注册钩子捕获长度模块的输出
        if hasattr(self.model, 'length_module'):
            hook = FeatureHook()
            handle = self.model.length_module.register_forward_hook(hook.hook_fn)
            self.hooks["length_module"] = (hook, handle)

    def _find_layer(self, name):
        """查找指定的层"""
        module = self.model
        parts = name.split(".")
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                if isinstance(module, nn.ModuleDict) and part in module:
                    module = module[part]
                else:
                    return None
        return module

    def register_hook(self, layer_path):
        target_layer = self._find_layer(layer_path)
        if target_layer is None:
            print(f"Warning: Layer {layer_path} not found")
            return self

        hook = FeatureHook()
        handle = target_layer.register_forward_hook(hook.hook_fn)
        self.hooks[layer_path] = (hook, handle)
        return self

    def remove_hooks(self):
        for _, handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()

    def preprocess_image(self, image_path):
        original_image = read_image(image_path, format="BGR")
        height, width = original_image.shape[:2]

        image = original_image.astype(np.float32)
        image = image[:, :, ::-1]  # BGR转RGB

        # 使用float32类型的均值和方差
        mean = np.array(self.cfg.MODEL.PIXEL_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.cfg.MODEL.PIXEL_STD, dtype=np.float32).reshape(1, 1, 3)
        image = (image - mean) / std

        image_tensor = torch.as_tensor(
            image.transpose(2, 0, 1),
            dtype=torch.float32
        )

        image_list = ImageList.from_tensors(
            [image_tensor.to(self.device)],
            self.size_divisibility
        )
        return original_image, image_list

    def visualize_features(self, features, output_dir, layer_name, original_image, prefix="",
                           heatmap_style=False, instance_idx=None):
        """特征可视化方法"""
        os.makedirs(output_dir, exist_ok=True)

        if features is None:
            return []

        # 处理不同维度的特征图
        if features.dim() == 4:  # [N, C, H, W]
            if instance_idx is not None and instance_idx < features.shape[0]:
                features = features[instance_idx]
            else:
                features = features[0]

        elif features.dim() == 1:  # [C]
            print(f"Skipping visualization for 1D feature vector (size: {features.shape})")
            return []

        if features.dim() < 2:
            features = features.unsqueeze(0)

        # 获取基本尺寸
        num_channels = features.shape[0]
        if features.dim() > 2:
            h_feat, w_feat = features.shape[-2], features.shape[-1]
        else:
            h_feat, w_feat = 1, features.shape[1]

        # 获取原始图像的宽度和高度
        height, width = original_image.shape[:2]

        # 为每个通道创建热力图
        output_paths = []
        for i in range(min(20, num_channels)):  # 最多可视化20个通道
            if features.dim() > 2:
                channel_data = features[i].numpy()
            else:
                channel_data = features[i].numpy().reshape(1, 1)

            # 归一化
            vmin = np.min(channel_data)
            vmax = np.max(channel_data)
            if vmax - vmin < 1e-6:
                channel_norm = np.ones_like(channel_data)
            else:
                channel_norm = (channel_data - vmin) / (vmax - vmin)

            # 转换为热力图
            heatmap = (channel_norm * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # 调整到原始图像尺寸
            heatmap = cv2.resize(heatmap, (width, height))

            # 叠加到原始图像上
            if heatmap_style and original_image is not None:
                heatmap = cv2.addWeighted(original_image, 0.5, heatmap, 0.7, 0)

            # 保存图像
            idx_text = f"_inst{instance_idx}" if instance_idx is not None else ""
            output_path = os.path.join(
                output_dir,
                f"{prefix}{layer_name}_ch{i:03d}{idx_text}.jpg"
            )
            cv2.imwrite(output_path, heatmap)
            output_paths.append(output_path)

        return output_paths

    def visualize_features_in_box(self, features, output_dir, original_image, pred_boxes,
                                  prefix="", instance_idx=None, channel_idx=None):
        """在检测框内显示特征图"""
        os.makedirs(output_dir, exist_ok=True)

        if features is None or len(pred_boxes) == 0:
            return []

        # 获取当前实例的检测框
        box = pred_boxes[instance_idx].tensor.cpu().numpy()[0]
        x1, y1, x2, y2 = box.astype(int)

        # 提取当前通道的特征
        channel_data = features[instance_idx, channel_idx].numpy()

        # 归一化
        vmin = np.min(channel_data)
        vmax = np.max(channel_data)
        if vmax - vmin < 1e-6:
            channel_norm = np.ones_like(channel_data)
        else:
            channel_norm = (channel_data - vmin) / (vmax - vmin)

        # 转换为热力图
        heatmap = (channel_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 调整到检测框尺寸
        box_h, box_w = y2 - y1, x2 - x1
        heatmap = cv2.resize(heatmap, (box_w, box_h))

        # 在检测框内显示特征图
        overlay = original_image.copy()
        if 0 <= y1 < y2 <= overlay.shape[0] and 0 <= x1 < x2 <= overlay.shape[1]:
            overlay[y1:y2, x1:x2] = cv2.addWeighted(
                overlay[y1:y2, x1:x2], 0.5,
                heatmap, 0.7, 0
            )

        # 绘制检测框
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 保存图像
        output_path = os.path.join(
            output_dir,
            f"{prefix}ch{channel_idx:03d}_inst{instance_idx}.jpg"
        )
        cv2.imwrite(output_path, overlay)

        return [output_path]

    def process_image(self, image_path, output_dir, heatmap_style=True):
        """处理单张图像 - 显示在检测框内"""
        original_img, image_list = self.preprocess_image(image_path)
        inputs = [{"image": image_list.tensor[0]}]

        with torch.no_grad():
            # 获取检测结果
            detector_output = self.model(inputs)[0]

        # 提取预测框
        instances = detector_output["instances"]
        pred_boxes = instances.pred_boxes

        # 提取增强特征
        output_paths = []
        length_module = self.model.length_module

        if hasattr(length_module, 'enhanced_features') and length_module.enhanced_features is not None:
            enhanced_features = length_module.enhanced_features.detach().cpu()

            # 遍历每个检测实例和每个通道
            for instance_idx in range(enhanced_features.shape[0]):
                for channel_idx in range(min(3, enhanced_features.shape[1])):
                    paths = self.visualize_features_in_box(
                        enhanced_features,
                        output_dir,
                        original_img,
                        pred_boxes,
                        prefix=os.path.basename(image_path).split('.')[0] + "_",
                        instance_idx=instance_idx,
                        channel_idx=channel_idx
                    )
                    output_paths.extend(paths)

        return output_paths


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
    parser.add_argument("--heatmap-style", action="store_true", help="使用热力图风格可视化")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_logger()
    cfg = setup_cfg(args)

    visualizer = FeatureVisualizer(cfg)
    visualizer.checkpointer.load(cfg.MODEL.WEIGHTS)

    if os.path.isfile(args.input):
        output_paths = visualizer.process_image(
            args.input,
            args.output_dir,
            heatmap_style=args.heatmap_style
        )
        print(f"成功生成 {len(output_paths)} 张特征可视化图像:")
        for path in output_paths:
            print(f" - {path}")
    else:
        print("处理整个目录中的图像...")
        all_images = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image in tqdm(all_images):
            image_path = os.path.join(args.input, image)
            try:
                visualizer.process_image(
                    image_path,
                    args.output_dir,
                    heatmap_style=args.heatmap_style
                )
            except Exception as e:
                print(f"处理图像 {image} 时出错: {str(e)}")

    visualizer.remove_hooks()


if __name__ == "__main__":
    main()