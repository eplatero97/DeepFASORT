TRAIN_REID = True
device = 'cuda'
_base_ = [
    '../_base_/datasets/mot_challenge_reid.py', '../_base_/default_runtime.py'
]

model = dict(
    reid=dict(
        type='BaseReID',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_pairwise=dict(
                type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'))
        ))
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[5])
total_epochs = 6


data_root = 'data/MOT17/'
data = dict(
    train=dict(
        data_prefix=data_root + 'reid/faimgs'),
    val=dict(
        data_prefix=data_root + 'reid/faimgs'),
    test=dict(
        data_prefix=data_root + 'reid/faimgs'))


proj_name = "train_fareid"
#wandb = dict(project="DeepFASORT", entity="eeplater", name=proj_name)
log_config = dict(
    interval=50,
    by_epoch=False,
    out_dir=proj_name,
    hooks=[
        dict(type='TextLoggerHook', out_dir=proj_name),
        dict(type='CheckpointHook', interval=1, by_epoch=True, out_dir=proj_name),
#        dict(type='WandbLoggerHook', init_kwargs=wandb)
    ])
