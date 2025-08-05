import sys
sys.path.append("/hy-tmp/BlendMask/detectron2")
sys.path.append("/hy-tmp/BlendMask/AdelaiDet")
import argparse
import os
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger

sys.path.insert(0, "/hy-tmp/BlendMask/AdelaiDet")
from adet.config import get_cfg as get_adet_cfg


def setup_cfg(config_path, weights_path):
    """动态配置加载方法"""
    cfg = get_adet_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.RPN.PRE_NMS_TOPK = 2000  # 控制显示的候选框密度
    cfg.freeze()
    return cfg


class RPNVisualizer:
    def __init__(self, metadata):
        self.metadata = metadata
        self.box_color = (200, 0, 0)  # 深红色BGR格式

    def visualize(self, image, proposals):
        """专用RPN候选框可视化方法"""
        visualizer = Visualizer(
            image[:, :, ::-1],  # BGR转RGB
            metadata=self.metadata,
            instance_mode=ColorMode.IMAGE_BW
        )

        # 绘制前200个高分候选框
        if len(proposals) > 200:
            keep = proposals.objectness_logits.topk(200).indices
            proposals = proposals[keep]

        visualizer.overlay_instances(
            boxes=proposals.proposal_boxes,
            assigned_colors=[self.box_color] * len(proposals),
            alpha=0.3  # 半透明效果
        )
        return visualizer.output


def extract_rpn_proposals(cfg, img_path, output_dir):
    """核心提取逻辑"""
    # 初始化模型
    from adet.modeling.blendmask import BlendMask  # 延迟导入确保路径正确
    # 图像读取与处理
    original_image = read_image(img_path, format="BGR")

    if not original_image.flags['C_CONTIGUOUS']:
        original_image = np.ascontiguousarray(original_image)
    original_image.flags.writeable = True  # 显式设为可写

    chw_image = original_image.transpose(2, 0, 1).copy()  # 添加.copy()

    image_tensor = torch.as_tensor(
        chw_image,
        dtype=torch.float32,
        device=cfg.MODEL.DEVICE
    ).unsqueeze(0)  # 添加batch维度 [1, C, H, W]

    # 打印验证信息
    print(f"输入张量形状验证: {image_tensor.shape}")
    print(f"张量设备位置: {image_tensor.device}")
    print(f"张量内存连续性: {image_tensor.is_contiguous()}")
    model = BlendMask(cfg)
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # 图像预处理
    original_image = read_image(img_path, format="BGR")
    image_tensor = torch.as_tensor(
        original_image.transpose(2, 0, 1),  # HWC -> CHW
        dtype=torch.float32
    ).unsqueeze(0).to(device)  # 添加batch维度 (NCHW)

    print(f"输入张量形状验证：{image_tensor.shape}")  # 应为 [1, 3, H, W]

    with torch.no_grad():
        features = model.backbone(image_tensor)
        proposals, _ = model.proposal_generator([image_tensor], features)

    # 可视化保存
    visualizer = RPNVisualizer(None)  # 不需要metadata
    vis_output = visualizer.visualize(original_image, proposals[0])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, vis_output.get_image()[:, :, ::-1])  # RGB转BGR保存


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--input", required=True, help="输入图像路径")
    parser.add_argument("--output", default="./rpn_output", help="输出目录")
    args = parser.parse_args()

    setup_logger()
    cfg = setup_cfg(args.config, args.weights)
    extract_rpn_proposals(cfg, args.input, args.output)
    print(f"可视化结果已保存至：{os.path.abspath(args.output)}")