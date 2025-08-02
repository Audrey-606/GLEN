import sys

sys.path.append("/hy-tmp/BlendMask/detectron2")
sys.path.append("/hy-tmp/BlendMask/AdelaiDet")
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
import torch.nn.functional as F
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from adet.config import get_cfg as get_adet_cfg


class BasisHook:
    def __init__(self):
        self.bases = None
        self.attentions = None  # 存储注意力图


class BasisVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.checkpointer = DetectionCheckpointer(self.model)
        self.hook = BasisHook()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.size_divisibility = self.model.backbone.size_divisibility

        # 仅注册基础特征钩子
        tower_layer = self.model.basis_module.tower
        self.handle_basis = tower_layer.register_forward_hook(
            lambda m, i, o: self.hook.__setattr__('bases', o.detach().cpu())
        )

    def preprocess_image(self, image_path):
        original_image = read_image(image_path, format="BGR")
        image = original_image.astype(np.float32)[:, :, ::-1]  # BGR转RGB

        # 标准化
        pixel_mean = np.array(self.cfg.MODEL.PIXEL_MEAN, dtype=np.float32).reshape(1, 1, 3)
        pixel_std = np.array(self.cfg.MODEL.PIXEL_STD, dtype=np.float32).reshape(1, 1, 3)
        image = (image - pixel_mean) / pixel_std

        image_tensor = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        image_list = ImageList.from_tensors([image_tensor.to(self.device)], self.size_divisibility)
        return original_image, image_list

    def visualize_bases(self, output_dir, prefix=""):
        os.makedirs(output_dir, exist_ok=True)
        if self.hook.bases is None:
            raise RuntimeError("No bases captured")

        bases = self.hook.bases[0]
        num_bases = bases.shape[0]

        output_paths = []
        for idx in range(num_bases):
            basis = bases[idx].numpy()
            basis = (basis - basis.min()) / (basis.max() - basis.min() + 1e-8)
            basis_uint8 = (basis * 255).astype(np.uint8)
            colored_basis = cv2.applyColorMap(basis_uint8, cv2.COLORMAP_HOT)

            filename = f"{prefix}basis_{idx:02d}.png"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, colored_basis)
            output_paths.append(output_path)
        return output_paths

    def visualize_length_attentions(self, attentions, output_dir, prefix=""):
        """可视化长度模块的注意力图"""
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for i, attn in enumerate(attentions):
            # 确保是3D张量 [1, H, W]
            if attn.dim() == 3:
                attn = attn[0]  # 去掉通道维度

            attn_np = attn.numpy()

            # 归一化
            v_min, v_max = attn_np.min(), attn_np.max()
            attn_norm = (attn_np - v_min) / (v_max - v_min + 1e-8)
            attn_uint8 = (attn_norm * 255).astype(np.uint8)

            # 应用颜色映射并保存
            jet_attn = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
            filename = f"{prefix}length_attn_{i:03d}.png"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, jet_attn)
            output_paths.append(output_path)

        return output_paths

    def visualize_attentions(self, output_dir, prefix=""):
        os.makedirs(output_dir, exist_ok=True)
        if self.hook.attentions is None:
            raise RuntimeError("No attentions captured")

        attns = self.hook.attentions
        K = self.cfg.MODEL.BASIS_MODULE.NUM_BASES
        M = self.cfg.MODEL.BLENDMASK.ATTN_SIZE
        H = self.cfg.MODEL.BLENDMASK.BOTTOM_RESOLUTION

        output_paths = []
        for inst_idx in range(attns.shape[0]):
            # 重塑为 K x M x M
            instance_attn = attns[inst_idx].view(K, M, M)

            # 插值到基础特征分辨率
            instance_attn = F.interpolate(
                instance_attn.unsqueeze(0).float(),
                size=H,
                mode=self.cfg.MODEL.BLENDMASK.TOP_INTERP
            ).squeeze(0).numpy()

            for k in range(K):
                attn_map = instance_attn[k]
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                attn_uint8 = (attn_map * 255).astype(np.uint8)
                colored_attn = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)

                filename = f"{prefix}inst_{inst_idx:03d}_attn_k{k:02d}.png"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, colored_attn)
                output_paths.append(output_path)
        return output_paths

    def process_image(self, image_path, output_dir):
        # 预处理
        _, image_list = self.preprocess_image(image_path)

        # 前向传播并捕获注意力
        with torch.no_grad():
            inputs = [{"image": image_list.tensor[0]}]
            predictions = self.model(inputs)

            # 直接从Blender模块获取注意力
            blender = self.model.blender
            if hasattr(blender, 'attns') and blender.attns is not None:
                self.hook.attentions = blender.attns.detach().cpu()
            else:
                print("Warning: No attentions captured in this forward pass")

        # 生成可视化
        basis_paths = self.visualize_bases(output_dir, prefix)
        prefix = os.path.basename(image_path).split('.')[0] + "_"
        all_paths = basis_paths
        #
        #if self.hook.attentions is not None:
        #    attn_paths = self.visualize_attentions(output_dir, prefix)
        #    all_paths = basis_paths + attn_paths
        #else:
        #    all_paths = basis_paths
        #prefix = os.path.basename(image_path).split('.')[0] + "_"

        # 提取所有实例的长度注意力
        all_length_attentions = []
        for result in predictions:
            if hasattr(result["instances"], 'pred_length_attn'):
                all_length_attentions.extend([attn for attn in result["instances"].pred_length_attn])

        if all_length_attentions:
            length_attn_paths = self.visualize_length_attentions(
                all_length_attentions,
                output_dir,
                prefix
            )
            all_paths.extend(length_attn_paths)

        return all_paths

    def __del__(self):
        self.handle_basis.remove()


def setup_cfg(args):
    cfg = get_adet_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, help="Model config file")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_logger()
    cfg = setup_cfg(args)

    visualizer = BasisVisualizer(cfg)
    visualizer.checkpointer.load(cfg.MODEL.WEIGHTS)

    output_paths = visualizer.process_image(args.input, args.output_dir)
    print(f"Saved {len(output_paths)} visualization files:")
    for path in output_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()