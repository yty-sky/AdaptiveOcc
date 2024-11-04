# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from termios import BS0
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16
import time

@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 temporal_num=1,
                 two_stage_num_proposals=300,
                 encoder=None,
                 embed_dims=256,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)

        self.encoder = build_transformer_layer_sequence(encoder)

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.temporal_num = temporal_num
        self.fp16_enabled = False
        # self.fp16_enabled = True
        self.use_cams_embeds = use_cams_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.temporal_embeds = None
        if self.temporal_num > 1:
            self.temporal_embeds = nn.Parameter(torch.Tensor(self.temporal_num, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'volume_queries'))
    def forward(
            self,
            mlvl_feats,
            volume_queries,
            volume_h,
            volume_w,
            volume_z,
            octree,
            depth,
            **kwargs):

        # 主要做了一件事：  嵌入位置特征和维度转换
        bs = mlvl_feats[0].size(0)
        volume_queries = volume_queries.unsqueeze(1).repeat(1, bs, 1)  # [10000, 1, 512]
        feat_flatten = []
        spatial_shapes = []

        for lvl, feat in enumerate(mlvl_feats):

            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            feat = feat.reshape(num_cam//self.temporal_num, self.temporal_num, h*w, c)#num_cam, bs, hw, c
            if self.use_cams_embeds:                                                       # 嵌入相机位置编码
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            if self.temporal_embeds is not None:
                feat = feat + self.temporal_embeds[None, :, None, :].to(feat.dtype)

            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)     # 嵌入特征尺寸位置编码
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)   # [6, 1, 1450, 512]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.int64, device=feat_flatten.device) # [29, 50] [58, 100] [116, 200] [232, 400]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])) # (num_cam, H*W, bs, embed_dims)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        volume_embed = self.encoder(
                volume_queries,
                feat_flatten,
                feat_flatten,
                octree,
                depth,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )

        return volume_embed


