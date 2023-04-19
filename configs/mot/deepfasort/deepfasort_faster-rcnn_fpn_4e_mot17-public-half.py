_base_ = ['./deepfasort_faster-rcnn_fpn_4e_mot17-private-half.py']


proj_name = "DeepSORT"
#wandb = dict(project="DeepFASORT", entity="eeplater", name=proj_name)
log_config = dict(
    interval=1,
    by_epoch=True,
    out_dir=proj_name,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='CheckpointHook'),
#        dict(type='WandbLoggerHook', init_kwargs=wandb)
    ])

model = dict(
    tracker=dict(
        preprocess_crop_cfg=None,
        preprocess_crop_checkpoint=None))
