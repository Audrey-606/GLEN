import sys
sys.path.append("/hy-tmp/BlendMask/detectron2")
sys.path.append("/hy-tmp/BlendMask/AdelaiDet")
import argparse
import os
import torch
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from predictor import VisualizationDemo
import multiprocessing as mp
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from adet.config import get_cfg
from yacs.config import CfgNode as CN


def setup_cfg(args):
    cfg = get_cfg()

    if not hasattr(cfg.MODEL, 'BLENDMASK'):
        cfg.MODEL.BLENDMASK = CN()
    cfg.MODEL.BLENDMASK.LENGTH_LOSS_WEIGHT = 0.5

    if not hasattr(cfg.MODEL, 'BASIS_MODULE'):
        cfg.MODEL.BASIS_MODULE = CN({'NUM_BASES': 4})

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # 配置评估参数
    cfg.DATASETS.TEST = ("seedlings_val",)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate BlendMask Model")
    parser.add_argument(
        "--config-file",
        default="configs/BlendMask/R_50_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input-dir",
        default="./input",
        help="path to directory with validation images",
    )
    parser.add_argument(
        "--ann-json",
        default="instances_val2017.json",
        help="path to COCO format annotation file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Minimum detection confidence threshold",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def register_dataset(input_dir, ann_file):
    for d in ["seedlings_val"]:
        if d in DatasetCatalog.list():
            DatasetCatalog.remove(d)

    # 注册验证集
    register_coco_instances(
        "seedlings_val",
        {},
        os.path.join(input_dir, ann_file),
        input_dir
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments:\n" + str(args))

    # 注册数据集
    register_dataset(args.input_dir, args.ann_json)

    # 配置模型
    cfg = setup_cfg(args)

    # 创建模型并加载权重
    model = VisualizationDemo(cfg).predictor.model

    # 构建数据加载器
    evaluator = COCOEvaluator("seedlings_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "eval"))
    val_loader = build_detection_test_loader(cfg, "seedlings_val")

    # 运行评估
    results = inference_on_dataset(model, val_loader, evaluator)


    if "segm" in results:
        logger.info("\n实例分割细节:")
        iou_list = results['segm']['IoU-array']
        logger.info(f"实例级平均IoU: {iou_list.mean():.3f}")
        logger.info(f"小目标IoU（面积<32²）: {iou_list[areas < 32 ** 2].mean():.3f}")


    logger.info("Evaluation results:")
    for task in ["segm"]:
        if f"{task}_AP" in results:
            logger.info(f"Task: {task.upper()}")
            logger.info(f"AP  : {results[f'{task}_AP']:.3f}")
            logger.info(f"AP50: {results[f'{task}_AP50']:.3f}")
            logger.info(f"AP75: {results[f'{task}_AP75']:.3f}")