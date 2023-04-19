_base_ = ['./deepfasort_faster-rcnn_fpn_4e_mot17-private-half.py']


proj_name = "DeepFASORT mmpose-selftrained fareid"
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
    reid=dict(
        init_cfg=dict(
            checkpoint='./work_dirs/resnet50_b32x8_faMOT17.pth' 
        )),
    tracker=dict(
        preprocess_crop_cfg='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_vdeepfasort.py',
        preprocess_crop_checkpoint='./work_dirs/hrnet_w48_coco_256x192_vdeepfasort.pth'))
