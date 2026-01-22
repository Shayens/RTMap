"""
Configuration for training RTMap using pre‑detected structural lane lines as
input.  This config disables camera and LiDAR sensing and instead relies
solely on lane detections provided by a custom dataset.  The detector
operates in ``lane`` modality and consumes ``external_lane_lines`` tensors.

Key changes from the original TbV config:

* ``input_modality`` disables all sensor inputs and enables external data.
* The model's ``modality`` is set to ``lane`` and image backbone/necks are
  removed since they are unused.
* The dataset uses :class:`LaneInputDataset` to generate noisy lane
  observations from the ground truth map.  Users should point
  ``map_info_file`` to the synthetic TbV annotations.

To launch training:

.. code:: bash

    ./tools/dist_train.sh projects/configs/rtmap/rtmap_lane_tbv.py 8

"""

_base_ = [
    '../datasets/custom_nus-3d.py',  # keep dataset registry definitions
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

num_map_classes = 3  # divider, ped_crossing, boundary
fixed_ptsnum_per_gt_line = 20

input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=True,
)

model = dict(
    type='RTMap',
    modality='lane',  # new modality for lane inputs
    use_grid_mask=False,
    video_test_mode=False,
    pretrained=None,
    img_backbone=None,  # no image backbone
    img_neck=None,
    pts_backbone=None,
    pts_neck=None,
    pts_bbox_head=dict(
        type='RTMapHead',
        bev_h=120,
        bev_w=240,
        num_query=900,
        num_vec_one2one=50,
        num_vec_one2many=0,
        k_one2many=0,
        num_pts_per_vec=fixed_ptsnum_per_gt_line,
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=dict(
            use_aux_seg=False
        ),
        z_cfg=dict(pred_z_flag=False, gt_z_flag=False),
        transformer=dict(
            type='MapTRPerceptionTransformer',
            num_cams=0,  # no cameras
            z_cfg=dict(pred_z_flag=False, gt_z_flag=False),
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
            encoder=None,  # encoder not used for lane input
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DecoupledDetrTransformerDecoderLayer',
                    num_vec=50,
                    num_pts_per_vec=fixed_ptsnum_per_gt_line,
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                        dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                        dict(type='CustomMSDeformableAttention', embed_dims=256, num_levels=1),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                ),
            ),
        ),
        bbox_coder=dict(
            type='RTMapNMSFreeCoder',
            z_cfg=dict(pred_z_flag=False, gt_z_flag=False),
            post_center_range=[-40, -25, 40, 25],
            pc_range=[-36.0, -18.0, 36.0, 18.0],
            max_num=50,
            voxel_size=[0.15, 0.15, 8.0],
            num_classes=num_map_classes,
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding', num_feats=128,
        ),
    ),
)

# Dataset settings: use LaneInputDataset for train/val/test
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='LaneInputDataset',
        map_info_file='data/tbv/tbv_map_infos_train.pkl',
        num_pts_per_lane=fixed_ptsnum_per_gt_line,
        noise_std=0.1,
        lane_shift_std=1.0,
        lane_shift_prob=0.1,
    ),
    val=dict(
        type='LaneInputDataset',
        map_info_file='data/tbv/tbv_map_infos_synthetic_val.pkl',
        num_pts_per_lane=fixed_ptsnum_per_gt_line,
        noise_std=0.0,
        lane_shift_std=0.0,
        lane_shift_prob=0.0,
        test_mode=True,
    ),
    test=dict(
        type='LaneInputDataset',
        map_info_file='data/tbv/tbv_map_infos_synthetic_val.pkl',
        num_pts_per_lane=fixed_ptsnum_per_gt_line,
        noise_std=0.0,
        lane_shift_std=0.0,
        lane_shift_prob=0.0,
        test_mode=True,
    ),
)

optimizer = dict(
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'lane_embed': dict(lr_mult=1.0)},
    ),
)

runner = dict(type='EpochBasedRunner', max_epochs=24)

lr_config = dict(policy='step', step=[16, 22])

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# Evaluation configuration can remain identical to original, focusing on
# localisation and change detection metrics.
