_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
occ_size = [25, 25, 2]
use_semantic = True


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names =  ['barrier','bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade','vegetation']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = [512, 256, 128]
_ffn_dim_ = [1024, 512, 256]

volume_h_ = 25
volume_w_ = 25
volume_z_ = 2

_num_points_ = [6, 4, 3]
_num_layers_ = [4, 3, 2]
_mlvl_feats_index = [2, 1, 0]

_build_octree = [2, 4, 5]
_build_octree_up = True

model = dict(
    type='AdaptiveOcc',
    use_grid_mask=True,
    use_semantic=use_semantic,
    img_backbone=dict(
       type='ResNet',
       depth=101,
       num_stages=4,
       out_indices=(1,2,3),
       frozen_stages=1,
       norm_cfg=dict(type='BN2d', requires_grad=False),
       norm_eval=True,
       style='caffe',
       dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
       stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='NormFPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        is_fpn=True,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='OccHead_DenseSkip',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=17,
        conv_input=_dim_,
        embed_dims=_dim_,
        img_channels=[512, 512, 512],
        use_semantic=use_semantic,
        is_gn=True,
        mlvl_feats_index=_mlvl_feats_index,
        is_train=False,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            num_cams=6,
            num_feature_levels=3,
            encoder=dict(
                type='OccEncoder',
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            num_cams=6,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
    ),
)

dataset_type = 'CustomNuScenesOccDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),  # 加载图片,并添加若干属性  17 + 7 = 24
    dict(type='PhotoMetricDistortionMultiViewImage'),           # 对图片做随机数据增强,不添加属性
    dict(type='LoadOccupancy', use_semantic=use_semantic),      # 加载occ真值，真值只包含被占用体素的坐标和语义,添加occ属性 24 + 1 = 25
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),       # 对图片做归一化，减去固定均值，除以1方差
    dict(type='PadMultiViewImage', size_divisor=32),            # 对图像做了pad: [900, 1600]->[928, 1600]  25 + 2 = 27
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),     #img打包成DataContainer
    dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img','gt_occ'])
]

find_unused_parameters = True
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        build_octree=_build_octree,
        build_octree_up=_build_octree_up,
        use_semantic=use_semantic,
        classes=class_names,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
         data_root=data_root,
         ann_file='data/nuscenes_infos_val.pkl',
         pipeline=test_pipeline,
         occ_size=occ_size,
         pc_range=point_cloud_range,
         build_octree=_build_octree,
         build_octree_up=_build_octree_up,
         use_semantic=use_semantic,
         classes=class_names,
         modality=input_modality),
    test=dict(type=dataset_type,
          data_root=data_root,
          ann_file='data/nuscenes_infos_val.pkl',
          pipeline=test_pipeline,
          occ_size=occ_size,
          pc_range=point_cloud_range,
          build_octree=_build_octree,
          build_octree_up=_build_octree_up,
          use_semantic=use_semantic,
          classes=class_names,
          modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.3),
        }),
    weight_decay=0.01)

optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=8, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=50, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=6)