import sys
import argparse
import os
import cv2
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from typing import List, Dict

sys.path.append("/hy-tmp/BlendMask/detectron2")
sys.path.append("/hy-tmp/BlendMask/AdelaiDet")

from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from adet.config import get_cfg
from predictor import VisualizationDemo


class BlendMaskGradCAM:
    def __init__(self, model, target_layer="backbone.res5.2.conv3"):
        self.model = model
        print("Backbone层结构：")
        print(model.backbone)
        self.activations = []
        self.gradients = []

        # 精确查找目标层
        self.target_layer = self._find_layer(target_layer)
        self._register_hooks()
        print(f"目标层结构：\n{self.target_layer}")

    def _find_layer(self, name):
        """支持ModuleDict的层查找方法"""
        module = self.model
        for part in name.split("."):
            if isinstance(module, nn.ModuleDict):
                module = module[part]
            else:
                module = getattr(module, part)
        return module

    def _register_hooks(self):
        """注册精确的前向/反向钩子"""

        def forward_hook(module, input, output):
            self.activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, image_tensor):
        """生成Grad-CAM热力图（修复字典问题）"""
        self.activations.clear()
        self.gradients.clear()

        # 前向传播
        outputs = self.model([{"image": image_tensor}])

        # 检查激活是否捕获
        if not self.activations:
            print("未捕获到目标层激活，请检查层名称")
            return np.zeros(image_tensor.shape[1:3], dtype=np.float32)

        # 反向传播梯度计算
        if len(outputs[0]['instances']) == 0:
            print("未检测到实例")
            return np.zeros(image_tensor.shape[1:3], dtype=np.float32)

        score = outputs[0]['instances'].scores[0]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # 处理激活和梯度
        activations = self.activations[0].cpu().numpy()[0]
        gradients = self.gradients[0].cpu().numpy()[0]

        # 计算权重和CAM
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # 后处理
        cam = np.maximum(cam, 0)
        target_width = image_tensor.shape[2]
        target_height = image_tensor.shape[1]
        cam = cv2.resize(cam, (target_width, target_height))
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-10)
        print(f"激活形状：{self.activations[0].shape}")
        return cam

class HeatmapVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata, instance_mode=None):
        super().__init__(img_rgb, metadata, instance_mode=instance_mode)
        # 定义配色
        self.color_stops = {
            'position': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            'colors': [
                (0, 0, 255),  # 蓝色 → 低值
                (0, 255, 255),  # 青色
                (255, 0, 0),  # 红色
                (255, 165, 0),  # 橙色
                (173, 216, 230),  # 浅蓝色
                (255, 255, 0)  # 黄色 → 高值
            ]
        }

        self.custom_colormap = self._create_sci_colormap()

    def _create_sci_colormap(self):
        """创建符合OpenCV要求的LUT（修复维度问题）"""
        # 转换为BGR格式
        bgr_colors = [np.array([b, g, r]) for r, g, b in self.color_stops['colors']]

        # 生成256级渐变
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        positions = np.array(self.color_stops['position']) * 255

        # 分段线性插值
        for i in range(256):
            for j in range(len(positions) - 1):
                if positions[j] <= i <= positions[j + 1]:
                    t = (i - positions[j]) / (positions[j + 1] - positions[j])
                    lut[i, 0] = (1 - t) * bgr_colors[j] + t * bgr_colors[j + 1]
                    break
        return lut

    def overlay_heatmap(self, cam, alpha=0.5):
        """Overlay heatmap on image"""
        # 确保输入在[0,1]范围
        cam = np.clip(cam, 0, 1)

        # 生成单通道热力图
        heatmap = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, self.custom_colormap)

        # 转换颜色空间用于叠加（BGR -> RGB）
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)
        # overlay = cv2.addWeighted(self.img, alpha, heatmap, 1 - alpha, 0)
        overlay = cv2.addWeighted(self.img, 0.5, heatmap, 0.5, 0)
        return overlay

    def generate_heatmap_variants(self, cam, instances):
        """Generate different heatmap visualization variants"""
        results = {}

        # Base heatmap
        results["base"] = self.overlay_heatmap(cam)

        # Box-restricted heatmap
        box_heatmap = np.zeros_like(self.img)
        if instances.has("pred_boxes"):
            for box in instances.pred_boxes:
                x1, y1, x2, y2 = map(int, box.cpu().numpy().squeeze())
                box_heatmap[y1:y2, x1:x2] = self.overlay_heatmap(cam)[y1:y2, x1:x2]
        results["box"] = box_heatmap

        # Red-filtered heatmap
        hsv = cv2.cvtColor(self.overlay_heatmap(cam), cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        results["filtered"] = cv2.bitwise_and(
            self.overlay_heatmap(cam),
            self.overlay_heatmap(cam),
            mask=mask
        )

        return results


class HeatmapVisualizationDemo(VisualizationDemo):
    def __init__(self, cfg, heatmap_layer="backbone.res5.2.conv3"):
        super().__init__(cfg)
        self.device = next(self.predictor.model.parameters()).device
        self.cam_generator = BlendMaskGradCAM(
            self.predictor.model,
            target_layer=heatmap_layer
        )

    def run_on_image(self, image):
        try:
            image = np.asarray(image, order='C', dtype=np.uint8).copy()
            print(f"原始图像形状: {image.shape}")

            if image.ndim == 2:  # 灰度图处理
                image = np.stack([image] * 3, axis=-1)
                print(f"灰度图扩展后形状: {image.shape}")

            elif image.shape[2] > 3:  # 处理RGBA等格式
                image = image[:, :, :3]
                print(f"去除alpha通道后形状: {image.shape}")

            chw_image = np.ascontiguousarray(image.transpose(2, 0, 1))  # HWC -> CHW
            print(f"CHW格式验证: {chw_image.shape} | 内存连续: {chw_image.flags['C_CONTIGUOUS']}")

            image_tensor = torch.from_numpy(chw_image).float()
            image_tensor = image_tensor.to(self.device)
            print(f"最终张量形状: {image_tensor.shape} | 设备: {image_tensor.device}")

        except Exception as e:
            print(f"图像预处理失败: {str(e)}")
            print(f"调试信息 - 输入类型: {type(image)} 形状: {getattr(image, 'shape', '无')}")
            raise

        # 生成预测和热力图
        predictions = self.predictor(image)
        cam = self.cam_generator.generate(image_tensor)

        # 可视化处理
        visualizer = HeatmapVisualizer(
            image[:, :, ::-1],  # BGR转RGB
            self.metadata
        )

        heatmaps = visualizer.generate_heatmap_variants(
            cam,
            predictions.get("instances", None)
        )

        return predictions, heatmaps


def setup_cfg(args):
    cfg = get_cfg()

    # Add BlendMask specific config
    if not hasattr(cfg.MODEL, 'BLENDMASK'):
        cfg.MODEL.BLENDMASK = CN()
    cfg.MODEL.BLENDMASK.LENGTH_LOSS_WEIGHT = 0.5

    if not hasattr(cfg.MODEL, 'BASIS_MODULE'):
        cfg.MODEL.BASIS_MODULE = CN({'NUM_BASES': 4})

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set score thresholds
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="BlendMask Heatmap Visualization")
    parser.add_argument(
        "--config-file",
        default="configs/BlendMask/R_50_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="Input images or directories",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/heatmaps",
        help="path to output directory",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions",
    )
    parser.add_argument(
        "--heatmap-layer",
        type=str,
        default="se_blocks.p3",
        help="Target layer for Grad-CAM",
    )
    parser.add_argument(
        "--heatmap-types",
        nargs="+",
        default=["base", "box", "filtered"],
        choices=["base", "box", "filtered"],
        help="Types of heatmaps to generate",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()
    setup_logger(name="heatmaps")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize demo
    demo = HeatmapVisualizationDemo(cfg, heatmap_layer=args.heatmap_layer)

    # Process inputs
    for path in args.input:
        # Read image
        img = read_image(path, format="BGR")

        # Run inference and generate heatmaps
        predictions, heatmaps = demo.run_on_image(img)

        # Save results
        filename = os.path.basename(path)
        for heatmap_type in args.heatmap_types:
            if heatmap_type in heatmaps:
                output_path = os.path.join(
                    args.output_dir,
                    f"{os.path.splitext(filename)[0]}_{heatmap_type}.jpg"
                )
                cv2.imwrite(output_path, heatmaps[heatmap_type][:, :, ::-1])  # RGB to BGR
                logger.info(f"Saved {heatmap_type} heatmap to {output_path}")


if __name__ == "__main__":
    main()