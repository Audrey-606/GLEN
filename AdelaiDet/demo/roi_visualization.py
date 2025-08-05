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


class FeatureVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.checkpointer = DetectionCheckpointer(self.model)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.size_divisibility = self.model.backbone.size_divisibility

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

    def generate_roi_images(self, image_path, output_dir):
        """为每个检测到的ROI生成单独的带框原图"""
        original_img, image_list = self.preprocess_image(image_path)
        inputs = [{"image": image_list.tensor[0]}]

        with torch.no_grad():
            # 获取检测结果
            detector_output = self.model(inputs)[0]

        # 提取预测框
        instances = detector_output["instances"]
        pred_boxes = instances.pred_boxes.tensor.cpu().numpy()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 为每个检测框生成图像
        output_paths = []
        for i, box in enumerate(pred_boxes):
            # 复制原始图像
            img_with_box = original_img.copy()

            # 转换坐标格式
            x1, y1, x2, y2 = map(int, box)

            # 在图像上绘制框
            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 保存图像
            filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{filename}_roi_{i}.jpg")
            cv2.imwrite(output_path, img_with_box)
            output_paths.append(output_path)

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
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_logger()
    cfg = setup_cfg(args)

    visualizer = FeatureVisualizer(cfg)
    visualizer.checkpointer.load(cfg.MODEL.WEIGHTS)

    if os.path.isfile(args.input):
        output_paths = visualizer.generate_roi_images(
            args.input,
            args.output_dir
        )
        print(f"成功生成 {len(output_paths)} 张ROI图像:")
        for path in output_paths:
            print(f" - {path}")
    else:
        print("处理整个目录中的图像...")
        all_images = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image in tqdm(all_images):
            image_path = os.path.join(args.input, image)
            try:
                visualizer.generate_roi_images(
                    image_path,
                    args.output_dir
                )
            except Exception as e:
                print(f"处理图像 {image} 时出错: {str(e)}")


if __name__ == "__main__":
    main()