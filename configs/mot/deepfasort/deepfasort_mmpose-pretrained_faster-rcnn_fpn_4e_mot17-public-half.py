_base_ = ['./deepfasort_faster-rcnn_fpn_4e_mot17-private-half.py']


proj_name = "DeepFASORT mmpose-pretrained"
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
        preprocess_crop_cfg='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_vdeepfasort.py',
        preprocess_crop_checkpoint='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'))
