_base_ = ['./deepfasort_faster-rcnn_fpn_4e_mot17-private-half.py']


proj_name = "DeepFASORT mmpose-selftrained"
#wandb = dict(project="DeepFASORT", entity="eeplater", name=proj_name)
log_config = dict(
    out_dir=proj_name,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='CheckpointHook'),
#        dict(type='WandbLoggerHook', init_kwargs=wandb)
    ])

model = dict(
    tracker=dict(
        preprocess_crop_cfg='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_vdeepfasort.py',
        preprocess_crop_checkpoint='./work_dirs/hrnet_w48_coco_256x192_vdeepfasort.pth'))

data_root = 'data/MOT17/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDetections'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 1088),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img', 'public_bboxes'])
        ])
]
data = dict(
    val=dict(
        detection_file=data_root + 'train_split/half-val_detections.pkl',
        pipeline=test_pipeline),
    test=dict(
        detection_file=data_root + 'train_split/half-val_detections.pkl',
        pipeline=test_pipeline))
