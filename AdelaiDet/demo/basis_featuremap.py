import sys
sys.path.append("/hy-tmp/BlendMask/detectron2")
sys.path.append("/hy-tmp/BlendMask/AdelaiDet")
import os
import cv2
import numpy as np
import torch
import argparse
from typing import List
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from adet.config import get_cfg as get_adet_cfg


class AttnVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.checkpointer = DetectionCheckpointer(self.model)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.size_divisibility = self.model.backbone.size_divisibility
        self.attn_size = cfg.MODEL.BLENDMASK.ATTN_SIZE
        self.num_bases = cfg.MODEL.BASIS_MODULE.NUM_BASES

    def preprocess_image(self, image_path: str) -> tuple:
        original_image = read_image(image_path, format="BGR")
        image = original_image.astype(np.float32)[:, :, ::-1]
        pixel_mean = np.array(self.cfg.MODEL.PIXEL_MEAN).reshape(1, 1, 3)
        pixel_std = np.array(self.cfg.MODEL.PIXEL_STD).reshape(1, 1, 3)
        image = (image - pixel_mean) / pixel_std
        image_tensor = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        image_list = ImageList.from_tensors([image_tensor.to(self.device)], self.size_divisibility)
        return original_image, image_list

    def visualize_attns(self, attns: torch.Tensor, original_img: np.ndarray, output_dir: str, prefix: str = "") -> List[
        str]:
        """修改后的可视化方法（包含原始图像叠加）"""
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        if attns is None:
            return output_paths

        # 预处理原始图像
        h, w = original_img.shape[:2]
        attn_resolution = (self.attn_size * 16, self.attn_size * 16)  # 假设下采样率是16
        original_img_resized = cv2.resize(original_img, attn_resolution)  # 调整到注意力图分辨率

        for instance_idx in range(attns.shape[0]):
            instance_attn = attns[instance_idx]
            attn_map = instance_attn.view(self.num_bases, self.attn_size, self.attn_size)

            for basis_idx in range(self.num_bases):
                # 生成热力图
                basis_attn = attn_map[basis_idx].numpy()
                basis_attn = (basis_attn - basis_attn.min()) / (basis_attn.max() - basis_attn.min() + 1e-8)
                heatmap = cv2.applyColorMap((basis_attn * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # 调整热力图到原始图像分辨率
                heatmap = cv2.resize(heatmap, attn_resolution)

                # 叠加原图轮廓
                overlay = cv2.addWeighted(original_img_resized, 0.5, heatmap, 0.5, 0)

                # 保存结果
                filename = f"{prefix}inst{instance_idx}_basis{basis_idx}.png"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, overlay)
                output_paths.append(output_path)

        return output_paths

    def process_image(self, image_path: str, output_dir: str) -> List[str]:
        """修改后的处理流程"""
        original_img, image_list = self.preprocess_image(image_path)

        with torch.no_grad():
            inputs = [{"image": image_list.tensor[0]}]
            _ = self.model(inputs)
            attns = self.model.blender.attns

        prefix = os.path.basename(image_path).split('.')[0] + "_"
        return self.visualize_attns(attns, original_img, output_dir, prefix)  # 传递原始图像

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
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    visualizer = AttnVisualizer(cfg)
    visualizer.checkpointer.load(cfg.MODEL.WEIGHTS)
    output_paths = visualizer.process_image(args.input, args.output_dir)
    print(f"Generated {len(output_paths)} attention maps:")
    for path in output_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()