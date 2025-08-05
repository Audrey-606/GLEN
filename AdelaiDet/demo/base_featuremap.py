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
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from adet.config import get_cfg as get_adet_cfg


class BasisHook:
    def __init__(self):
        self.bases = None

    def hook_fn(self, module, input, output):
        self.bases = output.detach().cpu()


class BasisVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.checkpointer = DetectionCheckpointer(self.model)
        self.hook = BasisHook()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.size_divisibility = self.model.backbone.size_divisibility

        # 注册钩子到basis_module的tower层
        tower_layer = self.model.basis_module.tower
        self.handle = tower_layer.register_forward_hook(self.hook.hook_fn)

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

        bases = self.hook.bases[0]  # 取第一个样本
        num_bases = bases.shape[0]

        # 为每个基单独生成图片
        output_paths = []
        for idx in range(num_bases):
            basis = bases[idx].numpy()

            # 归一化到[0,1]
            basis = (basis - basis.min()) / (basis.max() - basis.min() + 1e-8)

            # 转换为uint8并应用颜色映射
            basis_uint8 = (basis * 255).astype(np.uint8)
            colored_basis = cv2.applyColorMap(basis_uint8, cv2.COLORMAP_VIRIDIS)

            # 生成文件名
            filename = f"{prefix}basis_{idx:02d}.png"
            output_path = os.path.join(output_dir, filename)

            # 保存图片
            cv2.imwrite(output_path, colored_basis)
            output_paths.append(output_path)

        return output_paths

    def process_image(self, image_path, output_dir):
        # 预处理
        _, image_list = self.preprocess_image(image_path)

        # 模型前向传播
        with torch.no_grad():
            inputs = [{"image": image_list.tensor[0]}]
            self.model(inputs)

        # 生成可视化
        prefix = os.path.basename(image_path).split('.')[0] + "_"
        return self.visualize_bases(output_dir, prefix)

    def __del__(self):
        self.handle.remove()  # 清理钩子


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
    print(f"Saved {len(output_paths)} basis images:")
    for path in output_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()