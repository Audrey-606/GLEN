import sys

sys.path.append("/hy-tmp/BlendMask/detectron2")
sys.path.append("/hy-tmp/BlendMask/AdelaiDet")
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
from adet.config import get_cfg
from yacs.config import CfgNode as CN
from skimage.morphology import skeletonize  # 添加骨架化库
from skimage.measure import label, regionprops  # 添加区域属性分析库

WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    cfg = get_cfg()

    if not hasattr(cfg.MODEL, 'BLENDMASK'):
        cfg.MODEL.BLENDMASK = CfgNode()
    cfg.MODEL.BLENDMASK.LENGTH_LOSS_WEIGHT = 0.5

    if not hasattr(cfg.MODEL, 'BASIS_MODULE'):
        cfg.MODEL.BASIS_MODULE = CfgNode(new_allowed=True)
        cfg.MODEL.BASIS_MODULE.NUM_BASES = 4

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # 设置置信度阈值
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/BlendMask/R_50_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input-dir",
        default="./input",
        help="path to input directory containing images",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/predicate",
        help="path to output directory",
    )
    parser.add_argument(
        "--ratio-file",
        default="pixel_to_mm_ratios.txt",
        help="name of the file containing pixel to mm ratios",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def load_pixel_to_mm_ratios(ratio_file_path):
    ratios = {}
    if os.path.exists(ratio_file_path):
        with open(ratio_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    try:
                        ratio = float(parts[1])
                        ratios[filename] = ratio
                    except ValueError:
                        continue
    return ratios


def calculate_skeleton_length(mask):
    """
    计算掩码骨架化后的像素长度
    """
    # 将掩码转换为布尔类型
    binary_mask = mask.astype(bool)

    # 骨架化处理
    skeleton = skeletonize(binary_mask)

    # 计算骨架的像素数量（即长度）
    length = np.sum(skeleton)

    return length


class MaskVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata, pixel_to_mm=1.0, instance_mode=None):
        super().__init__(img_rgb, metadata, instance_mode=instance_mode)
        self.pixel_to_mm = pixel_to_mm

    def draw_instance_predictions(self, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else []
        masks = predictions.pred_masks.numpy() if predictions.has("pred_masks") else None

        # 生成带骨架化长度的标签
        labels = []
        if predictions.has("pred_masks"):
            for i, mask in enumerate(masks):
                # 计算当前实例的骨架化长度
                skeleton_length = calculate_skeleton_length(mask)

                # 获取类别名称
                cls_name = self.metadata.thing_classes[classes[i]]

                # 创建标签：类别名称 + 置信度 + 骨架长度(px)
                label_str = f"{cls_name} {scores[i]:.2f} ({skeleton_length:.1f}px)"
                labels.append(label_str)

        return self.overlay_instances(
            boxes=boxes,
            masks=masks,
            labels=labels,
            alpha=0.6
        )


class MaskVisualizationDemo(VisualizationDemo):
    def __init__(self, cfg, pixel_to_mm=1.0):
        super().__init__(cfg)
        self.pixel_to_mm = pixel_to_mm

    def run_on_image(self, image):
        predictions = self.predictor(image)
        image = image[:, :, ::-1]  # BGR转RGB

        visualizer = MaskVisualizer(
            image,
            self.metadata,
            pixel_to_mm=self.pixel_to_mm,
            instance_mode=self.instance_mode
        )

        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")

            # 直接使用模型输出的掩膜
            if instances.has("pred_masks"):
                # 获取掩码数据
                masks = instances.pred_masks.numpy()

                # 调整掩膜尺寸匹配原图（如果需要）
                new_masks = []
                for mask in masks:
                    # 仅当掩码尺寸不匹配时才调整
                    if mask.shape != image.shape[:2]:
                        resized_mask = cv2.resize(
                            mask.astype(float),
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                        new_mask = resized_mask > 0.5
                    else:
                        new_mask = mask
                    new_masks.append(new_mask)

                # 更新实例的掩码
                instances.pred_masks = torch.from_numpy(np.stack(new_masks))

        vis_output = visualizer.draw_instance_predictions(instances)
        return predictions, vis_output


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments:\n" + str(args))

    cfg = setup_cfg(args)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载像素到毫米转换比例
    ratio_file_path = os.path.join(args.input_dir, args.ratio_file)
    pixel_to_mm_ratios = load_pixel_to_mm_ratios(ratio_file_path)
    default_ratio = 0.05  # 默认比例

    # 初始化可视化器
    demo = MaskVisualizationDemo(cfg, pixel_to_mm=default_ratio)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))

    if not image_files:
        logger.warning(f"No images found in input directory: {args.input_dir}")
        exit(1)

    for image_path in tqdm.tqdm(image_files, desc="Processing images"):
        # 获取当前图片的pixel_to_mm比例
        filename = os.path.basename(image_path)
        pixel_to_mm = pixel_to_mm_ratios.get(filename, default_ratio)

        # 更新demo的比例
        demo.pixel_to_mm = pixel_to_mm

        # 处理图片
        img = read_image(image_path, format="BGR")
        _, vis_output = demo.run_on_image(img)

        # 保存结果
        output_path = os.path.join(args.output_dir, filename)
        vis_output.save(output_path)

    logger.info(f"Processing complete. Results saved to {args.output_dir}")