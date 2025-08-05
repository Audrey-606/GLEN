import torch
import torch.nn.functional as F
from torch import nn
from detectron2.structures import ImageList, Boxes, Instances
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling import (
    build_backbone, build_proposal_generator,
    build_sem_seg_head, META_ARCH_REGISTRY
)
from detectron2.modeling.poolers import ROIPooler
from .blender import build_blender
from .basis_module import build_basis_module

__all__ = ["BlendMask"]


class LengthPredictionModule(nn.Module):
    def __init__(self, cfg, in_channels=256):
        super().__init__()
        # 从配置获取参数
        self.num_bases = cfg.MODEL.BASIS_MODULE.NUM_BASES
        self.attn_size = cfg.MODEL.BLENDMASK.ATTN_SIZE
        self.pooler_resolution = 7  # 保持与ROI尺寸一致
        self.shared_base = None

        self.conv_layers = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持空间维度
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.length_attention= nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self._init_weights()
        self.enhanced_features = None  # 存储空间注意力增强后的特征
        self.conv_features = None  # 存储卷积层后的特征
        self.final_features = None  # 存储全局平均池化前的特征
        self.spatial_attentions = []  # 存储每个实例的注意力图
        self.current_attentions = None  # 临时存储当前批次的注意力图

    def _init_weights(self):
        # 卷积层初始化
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)

        # 全连接层初始化
        nn.init.kaiming_normal_(self.fc[0].weight, mode='fan_out')
        nn.init.constant_(self.fc[0].bias, 0.1)
        nn.init.normal_(self.fc[-1].weight, std=0.001)
        nn.init.constant_(self.fc[-1].bias, 0.01)

        for layer in self.length_attention:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, features, instances):
        if not instances:
            return torch.tensor([], device=features['p3'].device)

        boxes = []
        for x in instances:
            if hasattr(x, 'pred_boxes'):
                boxes.append(x.pred_boxes)
            elif hasattr(x, 'proposal_boxes'):
                boxes.append(x.proposal_boxes)
            else:
                boxes.append(x.gt_boxes)

        roi_pooler = ROIPooler(
            output_size=7,
            scales=(0.125,),
            sampling_ratio=2,
            pooler_type="ROIAlign"
        )

        features_p3 = [features['p3']]
        box_features = roi_pooler(features_p3, boxes)  # [N,256,7,7]
        if self.shared_base:
            base_features = self.shared_base(box_features)
        else:
            # 没有共享层时，使用原始特征
            base_features = box_features

        # 清空前一次的结果
        self.spatial_attentions = []

        # 生成长度预测专用注意力
        spatial_attn = self.length_attention(base_features)  # [N,1,7,7]

        # 保存每个实例的注意力图
        for i in range(spatial_attn.shape[0]):
            # 分离、转换为cpu并保存
            self.spatial_attentions.append(spatial_attn[i:i + 1].detach().cpu())

        # 注意力增强
        enhanced_features = base_features * spatial_attn

        # 卷积特征提取
        conv_out = self.conv_layers(enhanced_features)
        self.conv_features = conv_out

        # 全局平均池化
        pooled = self.global_pool(conv_out)  # [N,64,1,1]
        self.final_features = conv_out
        pooled = pooled.view(pooled.size(0), -1)  # [N,64]


        # 全连接回归
        return self.fc(pooled).squeeze(1)  # [N]

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


@META_ARCH_REGISTRY.register()
class BlendMask(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        # 核心组件初始化
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.blender = build_blender(cfg)
        self.basis_module = build_basis_module(cfg, self.backbone.output_shape())
        self.length_attentions = []

        # SE模块初始
        self.se_blocks = nn.ModuleDict()
        fpn_channels = cfg.MODEL.FPN.OUT_CHANNELS
        for l in ['p3', 'p4', 'p5', 'p6', 'p7']:
            self.se_blocks[l] = SEBlock(fpn_channels)
        self.shared_attn_base = nn.Sequential(
            nn.Conv2d(fpn_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.length_module = LengthPredictionModule(cfg, in_channels=fpn_channels)
        self.length_loss_weight = cfg.MODEL.BLENDMASK.get("LENGTH_LOSS_WEIGHT", 0.1)  # 使用get方法提供默认值
        self.length_module.shared_base = self.shared_attn_base
        # Basis attention层
        num_bases = cfg.MODEL.BASIS_MODULE.NUM_BASES
        attn_size = cfg.MODEL.BLENDMASK.ATTN_SIZE

        self.top_layer = nn.Conv2d(
            fpn_channels,
            num_bases * attn_size * attn_size,
            kernel_size=3,
            padding=1
        )
        nn.init.normal_(self.top_layer.weight, std=0.01)
        nn.init.constant_(self.top_layer.bias, 0)

        # 全景分割组件
        self.sem_seg_head = build_sem_seg_head(cfg,
                                               self.backbone.output_shape()) if cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED else None

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        # 图像预处理
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        # 特征提取
        backbone_features = self.backbone(images.tensor)
        features = {k: self.se_blocks[k](v) for k, v in backbone_features.items() if k in self.se_blocks}

        # Basis模块
        if self.training:
            # 训练模式保持原逻辑
            basis_sem = [x.get("basis_sem", torch.empty(0)).to(self.device) for x in batched_inputs]
        else:
            # 推理模式自动生成全零basis_sem
            basis_sem = [
                torch.zeros((1, x["image"].shape[1], x["image"].shape[2]), device=self.device)
                for x in batched_inputs
            ]

        basis_sem = ImageList.from_tensors(
            basis_sem,
            self.backbone.size_divisibility,
            0
        ).tensor

        basis_out, basis_losses = self.basis_module(features, basis_sem)

        # 生成提案
        gt_instances = [x.get("instances").to(self.device) for x in batched_inputs] if "instances" in batched_inputs[
            0] else None
        proposals, proposal_losses = self.proposal_generator(
            images,
            features,
            gt_instances,
            self.top_layer  # 传递attention层
        )

        # Blender处理
        detector_results, detector_losses = self.blender(
            basis_out["bases"],
            proposals,
            gt_instances
        )

        # 长度预测
        self.length_attentions = []
        length_input = gt_instances if self.training else detector_results
        length_preds = self.length_module(features, length_input)
        self.length_attentions = self.length_module.spatial_attentions


        if self.training:
            losses = {}
            # 合并所有损失
            losses.update({"loss_" + k: v for k, v in basis_losses.items()})
            losses.update({k: v * self.length_loss_weight for k, v in detector_losses.items()})
            losses.update(proposal_losses)

            # 计算长度损失
            if len(gt_instances) > 0:
                gt_lengths = torch.cat([x.gt_pixel_length for x in gt_instances], dim=0)
                if isinstance(length_preds, (list, tuple)):
                    pred_tensor = torch.cat(length_preds)
                else:
                    pred_tensor = length_preds
                length_loss = F.smooth_l1_loss(pred_tensor, gt_lengths)
                losses["loss_length"] = length_loss * self.length_loss_weight

            # 语义分割损失
            if self.sem_seg_head is not None:
                sem_gt = [x["sem_seg"].to(self.device) for x in batched_inputs]
                sem_gt = ImageList.from_tensors(sem_gt, self.backbone.size_divisibility, 255).tensor
                _, sem_loss = self.sem_seg_head(features, sem_gt)
                losses.update(sem_loss)

            return losses

        # 推理阶段
        processed_results = []
        for i, (result, input_dict, image_size) in enumerate(zip(detector_results, batched_inputs, images.image_sizes)):
            # 后处理参数
            height = input_dict.get("height", image_size[0])
            width = input_dict.get("width", image_size[1])

            if not hasattr(result, 'pred_masks'):
                try:
                    #调整basis维度顺序为 [C, H, W]
                    bases = basis_out["bases"][i].permute(1, 2, 0)  # 从 [C, H, W] -> [H, W, C]
                    instance_attn = result.top_feats  # [N, K]

                    # sum(C * K) over C dimension
                    masks = torch.einsum('hwc,nk->nhw', bases, instance_attn).sigmoid()
                    result.pred_masks = masks
                except Exception as e:
                    print(f"Mask generation failed: {str(e)}")
                    continue


            def custom_postprocess(instances, output_height, output_width, mask_threshold=0.5):
                instances = detector_postprocess(instances, output_height, output_width)
                # 掩码处理
                if hasattr(instances, 'pred_masks'):
                    # 调整掩码尺寸
                    if instances.pred_masks.shape[-2:] != (output_height, output_width):
                        # print(f"Resizing masks from {instances.pred_masks.shape} to [{output_height}, {width}]")
                        instances.pred_masks = F.interpolate(
                            instances.pred_masks.unsqueeze(1).float(),
                            size=(output_height, output_width),
                            mode="bilinear"
                        ).squeeze(1)

                    # 二值化
                    instances.pred_masks = instances.pred_masks > mask_threshold
                    instances.pred_masks = instances.pred_masks.to(torch.bool)

                return instances

            result = custom_postprocess(result, height, width)

            if len(length_preds) > 0:
                if isinstance(length_preds, list):
                    result.pred_lengths = length_preds[i]
                else:
                    split_sizes = [len(r.pred_boxes) for r in detector_results]
                    result.pred_lengths = torch.split(length_preds, split_sizes)[i]

            # 全景分割处理
            if self.sem_seg_head is not None:
                sem_seg = sem_seg_postprocess(
                    basis_out["sem_seg"][i],
                    image_size,
                    height,
                    width
                )
                result.sem_seg = sem_seg.argmax(dim=0)

            if self.length_attentions:
                start_idx = 0 if i == 0 else sum(len(res) for res in detector_results[:i])
                end_idx = start_idx + len(result)
                result.pred_length_attn = torch.cat(self.length_attentions[start_idx:end_idx])

            processed_results.append({"instances": result})

        return processed_results
