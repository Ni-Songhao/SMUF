# Dataset config for CX dual-frequency random 1:1:1 branch exposure.
#
# Branch convention:
# - branch0: C band
# - branch1: X band
#
# Important:
# This config loads complete C+X inputs.  The random single/dual-band exposure
# is performed inside ``LSKNet_Base`` via ``probs=[1, 1, 1]`` so that missing
# branches are replaced by prompt features consistently.

crop_size = (512, 512)

pack_meta_keys = (
    'img_path',
    'seg_map_path',
    'ori_shape',
    'img_shape',
    'pad_shape',
    'scale_factor',
    'flip',
    'flip_direction',
    'reduce_zero_label',
    'modality',
    'mask',
    'source_id',
    'source_name',
)

train_pipeline = [
    dict(
        type='LoadDualNpyWithRandomBandMask',
        probabilities=(0, 0, 1),
        source_id=1,
        source_name='CX'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1024, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=180, pad_val=0, seg_pad_val=255),
    dict(type='PackSegInputs', meta_keys=pack_meta_keys),
]

test_pipeline = [
    dict(
        type='LoadDualNpyWithRandomBandMask',
        probabilities=(0, 0, 1),
        source_id=1,
        source_name='CX'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs', meta_keys=pack_meta_keys),
]

dataset_cx_train = dict(
    type='MDSAR7',
    data_root='data/mpol-XC_mdsar7_filtered_mmseg',
    data_prefix=dict(
        img_path='img_dir_C/train',
        img_path2='img_dir_X/train',
        seg_map_path='ann_dir/train'),
    pipeline=train_pipeline)

dataset_cx_val = dict(
    type='MDSAR7',
    data_root='data/mpol-XC_mdsar7_filtered_mmseg',
    data_prefix=dict(
        img_path='img_dir_C/val',
        img_path2='img_dir_X/val',
        seg_map_path='ann_dir/val'),
    pipeline=test_pipeline)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dataset_cx_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dataset_cx_val)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
