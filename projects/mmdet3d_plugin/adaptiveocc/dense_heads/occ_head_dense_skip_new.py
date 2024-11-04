# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from projects.mmdet3d_plugin.surroundocc.loss.loss_utils import multiscale_supervision, geo_scal_loss, AdaptiveLoss, sem_scal_loss, FocalLoss
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet.models.utils import build_transformer
from mmcv.cnn.utils.weight_init import constant_init
import os
from torch.autograd import Variable
import ocnn
from ocnn.octree import Octree
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

@HEADS.register_module()
class OccHead_DenseSkip(nn.Module):
    def __init__(self,
                 *args,
                 transformer_template=None,
                 num_classes=6,
                 volume_h=25,
                 volume_w=25,
                 volume_z=4,
                 depth=4,
                 conv_input=None,
                 conv_output=None,
                 embed_dims=None,
                 img_channels=None,
                 use_semantic=True,
                 is_train=True,
                 is_gn=False,
                 is_weight=True,
                 mlvl_feats_index=[2, 1, 0],
                 is_aux=True,
                 is_short=True,
                 **kwargs):
        super(OccHead_DenseSkip, self).__init__()

        self.conv_input = conv_input
        self.conv_output = conv_output
        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.img_channels = img_channels
        self.use_semantic = use_semantic
        self.embed_dims = embed_dims
        self.fpn_level = len(self.embed_dims)
        self.transformer_template = transformer_template
        self.mlvl_feats_index = mlvl_feats_index
        self.is_gn = is_gn
        self._init_layers()
        self.octree = Octree(depth=depth, pc_range=[-10, -10, -2.6, 10, 10, 0.6], occ_size=[volume_h, volume_w, volume_z])
        self.octree.construct_neigh(0)
        self.is_train = is_train
        self.aux = is_aux
        self.is_weight = is_weight
        # self.fp16_enabled = True
        self.fp16_enabled = False

        if self.is_weight:
            self.class_weight_aux = torch.FloatTensor([1, 2, 1, 1, 1, 5])
            self.class_weight_aux = self.class_weight_aux.cuda()
            # dist_low1, dist_high1 = distance_table[1], distance_table[2]
            # self.mask_A = torch.ones((25, 25, 4))
            # self.mask_A[dist_low1:dist_high1, dist_low1:dist_high1, :] = 3
            # self.mask_A = self.mask_A.flatten()

    def _init_layers(self):
        self.transformer = nn.ModuleList()
        for i in range(self.fpn_level):
            transformer = copy.deepcopy(self.transformer_template)

            transformer.embed_dims = transformer.embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points[i]

            transformer.encoder.transformerlayers.feedforward_channels = \
                self.transformer_template.encoder.transformerlayers.feedforward_channels[i]
            
            transformer.encoder.transformerlayers.embed_dims = \
                self.transformer_template.encoder.transformerlayers.embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].embed_dims[i]
            
            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims[i]
            
            transformer.encoder.num_layers = self.transformer_template.encoder.num_layers[i]

            if i == 0:
                transformer.encoder.transformerlayers.is_octree = False
            else:
                transformer.encoder.transformerlayers.is_octree = True

            transformer.encoder.transformerlayers.occ_size = [self.volume_h, self.volume_w, self.volume_z]

            transformer_i = build_transformer(transformer)
            self.transformer.append(transformer_i)

        in_channels = self.conv_input

        conv_cfg = dict(type='Conv3d', bias=False)
        norm_cfg = dict(type='GN', num_groups=1, requires_grad=True)
        self.occ = nn.ModuleList()
        self.occ_aux = nn.ModuleList()

        if self.use_semantic:
            if self.is_gn:
                conv3d_layer1 = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[0],
                    out_channels=int(in_channels[0]/2),  # 设置为： 0：empty 1-16:语义类  17: not well
                    kernel_size=3,
                    stride=1,
                    padding=1)
                conv3d_layer2 = build_conv_layer(
                    conv_cfg,
                    in_channels=int(in_channels[0]/2),
                    out_channels=self.num_classes + 1,  # 设置为： 0：empty 1-16:语义类  17: not well
                    kernel_size=1,
                    stride=1,
                    padding=0)
                occ = nn.Sequential(conv3d_layer1, build_norm_layer(norm_cfg, int(in_channels[0]/2))[1], nn.ReLU(inplace=True), \
                                    conv3d_layer2, build_norm_layer(norm_cfg, self.num_classes + 1)[1], nn.ReLU(inplace=True))
                
                self.occ.append(occ)

                for i in range(1, self.fpn_level):
                    mid_channels = int(in_channels[i] / 2)
                    occ = nn.ModuleList([ocnn.modules.OctreeConvGnRelu(in_channels[i], mid_channels, group=8, nempty=True),
                                         ocnn.modules.OctreeConvGnRelu(mid_channels, self.num_classes + 1, group=1, nempty=True)])
                    self.occ.append(occ)

                i = self.fpn_level
                mid_channels = 32
                occ = nn.ModuleList([ocnn.modules.OctreeConvGnRelu(64, mid_channels, group=8, nempty=True),
                                     ocnn.modules.OctreeConvGnRelu(mid_channels, self.num_classes, group=1, nempty=True)])
                self.occ.append(occ)

                # --------------- occ_aux ----------------
                conv3d_layer1 = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[0],
                    out_channels=int(in_channels[0]/2),  # 设置为： 0：empty 1-16:语义类  17: not well
                    kernel_size=3,
                    stride=1,
                    padding=1)
                conv3d_layer2 = build_conv_layer(
                    conv_cfg,
                    in_channels=int(in_channels[0]/2),
                    out_channels=self.num_classes,  # 设置为： 0：empty 1-16:语义类  17: not well
                    kernel_size=1,
                    stride=1,
                    padding=0)
                occ = nn.Sequential(conv3d_layer1, build_norm_layer(norm_cfg, int(in_channels[0]/2))[1], nn.ReLU(inplace=True), \
                                    conv3d_layer2, build_norm_layer(norm_cfg, self.num_classes)[1], nn.ReLU(inplace=True))
                self.occ_aux.append(occ)

                for i in range(1, self.fpn_level):
                    mid_channels = int(in_channels[i] / 2)
                    occ = nn.ModuleList([ocnn.modules.OctreeConvGnRelu(in_channels[i], mid_channels, group=8, nempty=True),
                                         ocnn.modules.OctreeConvGnRelu(mid_channels, self.num_classes, group=1, nempty=True)])
                    self.occ_aux.append(occ)

                i = self.fpn_level
                mid_channels = 32
                occ = nn.ModuleList([ocnn.modules.OctreeConvGnRelu(64, mid_channels, group=8, nempty=True),
                                     ocnn.modules.OctreeConvGnRelu(mid_channels, self.num_classes, group=1, nempty=True)])
                self.occ_aux.append(occ)

            # else:
            #     occ = build_conv_layer(
            #         conv_cfg,
            #         in_channels=in_channels[0],
            #         out_channels=self.num_classes + 1,  # 设置为： 0：empty 1-16:语义类  17: not well
            #         kernel_size=1,
            #         stride=1,
            #         padding=0)
            #     self.occ.append(occ)

            #     for i in range(1, self.fpn_level):
            #         mid_channels = int(in_channels[i] / 2)
            #         occ = nn.ModuleList([ocnn.modules.OctreeConvBnRelu(in_channels[i], mid_channels, nempty=True),
            #                              ocnn.modules.OctreeConvBnRelu(mid_channels, self.num_classes + 1, nempty=True)])
            #         self.occ.append(occ)

            #     i = self.fpn_level
            #     mid_channels = int(in_channels[i] / 2)
            #     occ = nn.ModuleList([ocnn.modules.OctreeConvBnRelu(in_channels[i], mid_channels, nempty=True),
            #                          ocnn.modules.OctreeConvBnRelu(mid_channels, self.num_classes, nempty=True)])
            #     self.occ.append(occ)

        self.volume_embedding = nn.Embedding(self.volume_h * self.volume_w * self.volume_z, self.embed_dims[0])
        self.transfer_depth = nn.Conv1d(in_channels=self.img_channels[0],out_channels=self.embed_dims[0],kernel_size=1,stride=1)

        self.transfer_conv = nn.ModuleList()
        conv_cfg=dict(type='Conv2d', bias=True)
        for i in range(self.fpn_level):
            transfer_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=self.img_channels[i],
                    out_channels=self.embed_dims[i],
                    kernel_size=1,
                    stride=1)
            transfer_block = nn.Sequential(transfer_layer, nn.ReLU(inplace=True))
            self.transfer_conv.append(transfer_block)

        self.channels = self.embed_dims

        if self.is_gn:
            self.upsample = nn.ModuleList([ocnn.modules.OctreeDeconvGnRelu(
                self.channels[d - 1], self.channels[d], group=32, kernel_size=[2], stride=2,
                nempty=False) for d in range(1, self.fpn_level)])
            self.upsample.append(ocnn.modules.OctreeDeconvGnRelu(
                self.channels[-1], 64, group=16, kernel_size=[2], stride=2,
                nempty=False))

        self.skip_conv = nn.ModuleList([ocnn.modules.Conv1x1(self.channels[d], self.channels[d]) for d in range(0, self.fpn_level)])
        self.skip_conv.append(ocnn.modules.Conv1x1(64, 64))

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for i in range(self.fpn_level):
            self.transformer[i].init_weights()
                
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @force_fp32(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, gt_occ=None, occ_feat=None):

        # torch.cuda.synchronize()
        # start_time = time.time()

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        volume_embed = []
        if occ_feat is not None:
            volume_queries = self.volume_embedding.weight.to(dtype) + self.transfer_depth(occ_feat).squeeze(0).permute(1, 0)
            # volume_queries = self.transfer_depth(occ_feat).squeeze(0).permute(1, 0)
        else:
            volume_queries = self.volume_embedding.weight.to(dtype)   # [10000, 512]

        self.octree = self.octree.cuda()

        volume_queries_list = []
        volume_queries_list.append(volume_queries)

        occ_preds = []
        occ_preds_aux = []

        for i in range(self.fpn_level):

            index = self.mlvl_feats_index[i]
            volume_h, volume_w, volume_z  = self.volume_h, self.volume_w, self.volume_z
            _, _, C, H, W = mlvl_feats[index].shape
            view_features = self.transfer_conv[i](mlvl_feats[index].reshape(bs*num_cam, C, H, W)).reshape(bs, num_cam, -1, H, W)

            volume_embed_i = self.transformer[i](
                [view_features],
                volume_queries,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                img_metas=img_metas,
                octree=self.octree,
                depth=i
            )

            # todo 可以再添加卷积
            if i == 0:
                volume_embed_i = volume_embed_i + self.skip_conv[i](volume_queries_list[i])
                volume_embed_i = volume_embed_i.reshape(bs, self.volume_h, self.volume_w, self.volume_z, -1).permute(0, 4, 1, 2, 3) # [1, 512, 50, 50, 4]
                occ_pred = self.occ[i](volume_embed_i)    # [1, 18, 50, 50, 4]
                occ_pred = occ_pred.permute(0, 2, 3, 4, 1).reshape(self.volume_h * self.volume_w * self.volume_z, -1) # [10000, 18]
                
                if self.is_train and self.aux:
                    occ_pred_aux = self.occ_aux[i](volume_embed_i)    # [1, 18, 50, 50, 4]
                    occ_pred_aux = occ_pred_aux.permute(0, 2, 3, 4, 1).reshape(self.volume_h * self.volume_w * self.volume_z, -1)

                volume_embed_i = volume_embed_i.permute(0, 2, 3, 4, 1).reshape(self.volume_h * self.volume_w * self.volume_z, -1)
            else:
                volume_embed_i = torch.squeeze(volume_embed_i, 0)
                volume_embed_i = volume_embed_i + self.skip_conv[i](volume_queries_list[i]) + self.skip_conv[i](self.upsample[i-1](volume_queries_list[i-1], self.octree, i-1))
                occ_pred_mid = self.occ[i][0](volume_embed_i, self.octree, i)
                occ_pred = self.occ[i][1](occ_pred_mid, self.octree, i)
                
                if self.is_train and self.aux:
                    occ_pred_mid_aux = self.occ_aux[i][0](volume_embed_i, self.octree, i)    # [1, 18, 50, 50, 4]
                    # occ_pred_aux = self.occ[i][1](occ_pred_mid_aux, self.octree, i)
                    occ_pred_aux = self.occ_aux[i][1](occ_pred_mid_aux, self.octree, i)

            occ_preds.append(occ_pred)
            if self.is_train and self.aux:
                occ_preds_aux.append(occ_pred_aux)

            if self.is_train == True:
                gt = gt_occ.gt[i]
                split = (gt == 6)
                self.octree.octree_split(split, i)  # 有一个特性，当全部为False时，分裂第一个
                self.octree.octree_grow_down(i + 1)
                volume_queries = self.upsample[i](volume_embed_i, self.octree, i)
                volume_queries_list.append(volume_queries)
            else:
                split = (torch.argmax(occ_pred, dim=-1) == 6)
                self.octree.octree_split(split, i)  # 有一个特性，当全部为False时，分裂第一个
                self.octree.octree_grow_down(i + 1)
                volume_queries = self.upsample[i](volume_embed_i, self.octree, i)
                volume_queries_list.append(volume_queries)

        i = i + 1
        volume_embed_i = volume_queries + self.skip_conv[i](self.upsample[i-1](volume_queries_list[i-1], self.octree, i-1))
        occ_pred_mid = self.occ[-1][0](volume_queries, self.octree, 3)
        occ_pred = self.occ[-1][1](occ_pred_mid, self.octree, 3)
        occ_preds.append(occ_pred)

        if self.is_train and self.aux:
            occ_pred_mid_aux = self.occ[-1][0](volume_queries, self.octree, 3)    # [1, 18, 50, 50, 4]
            occ_pred_aux = self.occ[-1][1](occ_pred_mid_aux, self.octree, 3)
            occ_preds_aux.append(occ_pred_aux)

        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"head: {end_time - start_time}")

        return occ_preds, self.octree, occ_preds_aux

    @force_fp32(apply_to=('preds_list'))
    def loss(self,
             gt_occ,
             preds_list,
             preds_list_aux,
             img_metas):

        # weight_CE = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1])
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
        if self.is_weight:
            criterion_aux = nn.CrossEntropyLoss(ignore_index=255, weight=self.class_weight_aux, reduction="none")
        else:
            criterion_aux = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
        # self.mask_A = self.mask_A.cuda()
        #  尝试添加样本权重，加大对某些样本的权重损失
        loss_dict = {}

        for i in range(len(preds_list)):
            pred = preds_list[i]
            gt = gt_occ.gt[i]
        
            if i == 0:
                # if self.is_short:
                #     loss_occ_i = 8 * (torch.mean(criterion_short(pred, gt.long())*self.mask_A) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
                # else:
                loss_occ_i = 8 * (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
            elif i == 1:
                loss_occ_i = 4 * (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
            elif i == 2:
                loss_occ_i = 2 * (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
            else:
                loss_occ_i = criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long())
            
            if self.aux:
                gt_aux = gt_occ.gt_aux[i]
                pred_aux = preds_list_aux[i]
                if i == 0:
                    # if self.is_short:
                    #     loss_occ_i_aux = 8 * (torch.mean(criterion(pred_aux, gt_aux.long())*self.mask_A) + sem_scal_loss(pred_aux, gt_aux.long()) + geo_scal_loss(pred_aux, gt_aux.long()))
                    # else:
                    loss_occ_i_aux = 8 * (criterion_aux(pred_aux, gt_aux.long()) + sem_scal_loss(pred_aux, gt_aux.long()) + geo_scal_loss(pred_aux, gt_aux.long()))
                elif i == 1:
                    loss_occ_i_aux = 4 * (criterion_aux(pred_aux, gt_aux.long()) + sem_scal_loss(pred_aux, gt_aux.long()) + geo_scal_loss(pred_aux, gt_aux.long()))
                elif i == 2:
                    loss_occ_i_aux = 2 * (criterion_aux(pred_aux, gt_aux.long()) + sem_scal_loss(pred_aux, gt_aux.long()) + geo_scal_loss(pred_aux, gt_aux.long()))
                else:
                    loss_occ_i_aux = criterion_aux(pred_aux, gt_aux.long()) + sem_scal_loss(pred_aux, gt_aux.long()) + geo_scal_loss(pred_aux, gt_aux.long())

            loss_dict['loss_occ_{}'.format(i)] = loss_occ_i
            if self.aux:
                loss_dict['loss_occ_aux{}'.format(i)] = loss_occ_i_aux

        return loss_dict


    # @force_fp32(apply_to=('preds_list'))
    # def loss(self,
    #          gt_occ,
    #          preds_list,
    #          preds_list_aux,
    #          img_metas):

    #     criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
    #     criterion_weight = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
    #     #  尝试添加样本权重，加大对某些样本的权重损失
    #     loss_dict = {}

    #     for i in range(len(preds_list)):
    #         pred = preds_list[i]
    #         gt = gt_occ.gt[i]
    #         pred_aux = preds_list_aux[i]
    #         gt_aux = gt_occ.gt_aux[i]

    #         if i == 0:
    #             sample_weight = torch.ones_like(gt, dtype=torch.float32)
    #             sample_weight[gt_aux == 5] = 25
    #             loss_weight = (criterion_weight(pred, gt.long()) * sample_weight).mean()
    #             loss_weight_aux = (criterion(pred_aux, gt_aux.long()) * sample_weight).mean()
    #             loss_occ_i = 8 * (loss_weight + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
    #             loss_occ_i_aux = 8 * (loss_weight_aux + sem_scal_loss(pred_aux, gt_aux.long()) + geo_scal_loss(pred_aux, gt_aux.long()))
    #         elif i == 1:
    #             loss_occ_i = 4 * (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
    #             loss_occ_i_aux = 4 * (criterion(pred_aux, gt_aux.long()) + sem_scal_loss(pred_aux, gt_aux.long()) + geo_scal_loss(pred_aux, gt_aux.long()))
    #         elif i == 2:
    #             loss_occ_i = 2 * (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))
    #             loss_occ_i_aux = 2 * (criterion(pred_aux, gt_aux.long()) + sem_scal_loss(pred_aux, gt_aux.long()) + geo_scal_loss(pred_aux, gt_aux.long()))
    #         else:
    #             loss_occ_i = criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long())
    #             loss_occ_i_aux = criterion(pred_aux, gt_aux.long()) + sem_scal_loss(pred_aux, gt_aux.long()) + geo_scal_loss(pred_aux, gt_aux.long())

    #         loss_dict['loss_occ_{}'.format(i)] = loss_occ_i
    #         loss_dict['loss_occ_aux{}'.format(i)] = loss_occ_i_aux

    #     return loss_dict