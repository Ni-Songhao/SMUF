# MMSegmentation Integration (Modern MMEngine Style)

This folder contains the files needed to add the random single/dual-band LSKNet recipe to a modern MMSegmentation project that uses MMEngine-style registries, configs, `BaseTransform`, and `SegDataSample`.

## 1. Copy Files

From this release folder, copy files into the same relative paths in your MMSegmentation repository.

```text
mmseg/datasets/transforms/random_band_mask.py
mmseg/models/backbones/LSKNet_Moe.py
mmseg/models/backbones/extra_module/__init__.py
mmseg/models/backbones/extra_module/MoE_share.py
mmseg/models/decode_heads/Unetfeed.py
mmseg/models/segmentors/joint_encoder_decoder.py
configs/_base_/datasets/cx_random111.py
configs/MoE/LSKNet_MoE_cx_random111.py
```

The included config also expects the standard MMSegmentation base configs:

```text
configs/_base_/default_runtime.py
configs/_base_/schedules/schedule_160k.py
```

If your target MMSegmentation repo already has them, you do not need to overwrite them.

## 2. Use With `custom_imports`

The provided config uses `custom_imports`, so editing `__init__.py` files is not required.

```python
custom_imports = dict(
    imports=[
        'mmseg.datasets.transforms.random_band_mask',
        'mmseg.models.backbones.LSKNet_Moe',
        'mmseg.models.decode_heads.Unetfeed',
        'mmseg.models.segmentors.joint_encoder_decoder',
    ],
    allow_failed_imports=False)
```

## 3. Train

After copying the files, run from the MMSegmentation repository root:

```bash
python tools/train.py configs/MoE/LSKNet_MoE_cx_random111.py
```

For distributed training, use the standard MMSegmentation launcher, for example:

```bash
bash tools/dist_train.sh configs/MoE/LSKNet_MoE_cx_random111.py 4
```

## 4. Dataset Path

Edit `configs/_base_/datasets/cx_random111.py` to match your local dataset path.

Default branch convention:

```text
branch0: C band
branch1: X band
```

The default data prefix is:

```python
data_root = 'data/mpol-XC_mdsar7_filtered_mmseg'
img_path = 'img_dir_C/train'
img_path2 = 'img_dir_X/train'
seg_map_path = 'ann_dir/train'
```

## 5. MMSegmentation Docs

Use the official MMSegmentation main/latest docs for framework usage. These links target the modern MMEngine-style API rather than the old 0.x API:

- Config system: https://mmsegmentation.readthedocs.io/en/main/user_guides/1_config.html
- Train and test: https://mmsegmentation.readthedocs.io/en/main/user_guides/4_train_test.html
- Add new modules: https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_models.html
- Data transforms: https://mmsegmentation.readthedocs.io/en/main/advanced_guides/transforms.html
- Runtime and `custom_imports`: https://mmsegmentation.readthedocs.io/en/main/advanced_guides/customize_runtime.html
