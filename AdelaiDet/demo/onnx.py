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
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import onnx

WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    cfg = get_cfg()

    # 添加BlendMask必要配置
    if not hasattr(cfg.MODEL, 'BLENDMASK'):
        cfg.MODEL.BLENDMASK = CN()
    cfg.MODEL.BLENDMASK.LENGTH_LOSS_WEIGHT = 0.5

    if not hasattr(cfg.MODEL, 'BASIS_MODULE'):
        cfg.MODEL.BASIS_MODULE = CN({'NUM_BASES': 4})

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
    parser.add_argument(
        "--export-onnx",
        type=str,
        default="",
        help="Export model to ONNX format at specified path"
    )
    return parser


class ONNXExporter:
    def __init__(self, cfg):
        self.cfg = cfg

        # 强制设置输入尺寸对齐
        cfg.INPUT.MIN_SIZE_TRAIN = (800,)  # 设置固定训练尺寸
        cfg.INPUT.MAX_SIZE_TRAIN = 1333
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333
        cfg.MODEL.BACKBONE.SIZE_DIVISIBILITY = 64  # 调整尺寸对齐参数

        self.model = build_model(cfg)
        self.model.eval()

        # 确保模型在CPU且为float32
        self.model = self.model.to(torch.device("cpu")).float()

        # 添加尺寸验证包装器
        class SizeValidator(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.size_divisibility = cfg.MODEL.BACKBONE.SIZE_DIVISIBILITY

            def forward(self, x):
                # 验证输入尺寸可被整除
                h, w = x.shape[2:]
                assert h % self.size_divisibility == 0, f"Height {h} must be divisible by {self.size_divisibility}"
                assert w % self.size_divisibility == 0, f"Width {w} must be divisible by {self.size_divisibility}"
                return self.model(x)

        self.model = SizeValidator(self.model)

    def preprocess_image(self, image_path):
        original_image = read_image(image_path, format="BGR")

        # 强制调整尺寸到可被整除
        h, w = original_image.shape[:2]
        new_h = (h // 64 + 1) * 64  # 根据SIZE_DIVISIBILITY调整
        new_w = (w // 64 + 1) * 64
        resized_image = cv2.resize(original_image, (new_w, new_h))

        image = torch.as_tensor(resized_image.astype("float32").transpose(2, 0, 1))
        mean = torch.tensor(self.cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        std = torch.tensor(self.cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        return ((image - mean) / std).unsqueeze(0).to(torch.device("cpu"))

    def preprocess_image(self, image_path):
        original_image = read_image(image_path, format="BGR")
        # 转换为tensor并标准化
        image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
        mean = torch.tensor(self.cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        std = torch.tensor(self.cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        image = (image - mean) / std
        return image.unsqueeze(0)  # 添加batch维度

    def export(self, onnx_path, sample_image_path):
        sample_input = self.preprocess_image(sample_image_path)

        with torch.no_grad():
            torch.onnx.export(
                self.wrapped_model,
                sample_input,
                onnx_path,
                opset_version=13,  # 升级到最新opset
                input_names=["input"],
                output_names=["boxes", "scores", "classes", "masks"],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "max_objects"},
                    "scores": {0: "max_objects"},
                    "classes": {0: "max_objects"},
                    "masks": {0: "max_objects", 2: "mask_height", 3: "mask_width"}
                },
                custom_opsets={"prim": 1},  # 添加自定义操作符支持
                verbose=False
            )

        # 验证导出的模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments:\n" + str(args))

    cfg = setup_cfg(args)

    # 模型导出逻辑
    if args.export_onnx:
        # 必须指定示例图片路径（使用输入目录中的第一张图片）
        sample_image = next(glob.iglob(os.path.join(args.input_dir, '*.jpg')), None)
        if sample_image is None:
            raise ValueError("No sample image found for ONNX export")

        # 初始化导出器
        exporter = ONNXExporter(cfg)

        # 加载模型权重
        checkpointer = DetectionCheckpointer(exporter.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        logger.info(f"Loaded weights from {cfg.MODEL.WEIGHTS}")

        # 导出模型
        exporter.export(args.export_onnx, sample_image)
        logger.info(f"ONNX model successfully exported to {args.export_onnx}")
        exit(0)

    # 原始推理流程
    os.makedirs(args.output_dir, exist_ok=True)
    ratio_file_path = os.path.join(args.input_dir, args.ratio_file)
    pixel_to_mm_ratios = load_pixel_to_mm_ratios(ratio_file_path)
    default_ratio = 0.05

    demo = MaskVisualizationDemo(cfg, pixel_to_mm=default_ratio)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))

    if not image_files:
        logger.warning(f"No images found in input directory: {args.input_dir}")
        exit(1)

    for image_path in tqdm.tqdm(image_files, desc="Processing images"):
        filename = os.path.basename(image_path)
        pixel_to_mm = pixel_to_mm_ratios.get(filename, default_ratio)
        demo.pixel_to_mm = pixel_to_mm
        img = read_image(image_path, format="BGR")
        _, vis_output = demo.run_on_image(img)
        output_path = os.path.join(args.output_dir, filename)
        vis_output.save(output_path)

    logger.info(f"Processing complete. Results saved to {args.output_dir}")