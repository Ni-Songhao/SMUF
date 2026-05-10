# Open-source example config for the legacy CX random 1:1:1 branch exposure
# recipe.  The dataset loads complete C+X inputs; random branch exposure and
# prompt completion are handled by ``LSKNet_Base``.

_base_ = [
    '../_base_/datasets/cx_random111.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

custom_imports = dict(
    imports=[
        'mmseg.datasets.transforms.random_band_mask',
        'mmseg.models.backbones.LSKNet_Moe',
        'mmseg.models.decode_heads.Unetfeed',
        'mmseg.models.segmentors.joint_encoder_decoder',
    ],
    allow_failed_imports=False)

find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='JointEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='LSKNet_Base',
        in_chans=8,
        probs=[1, 1, 1],
        embed_dims=[64, 128, 320, 512],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    decode_head=dict(
        type='UnetFeedHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(256, 256)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0006,
        betas=(0.9, 0.999),
        weight_decay=0.01))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=320000,
        by_epoch=False),
]

iters = 320000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='mIoU',
        rule='greater'))

randomness = dict(seed=42)
