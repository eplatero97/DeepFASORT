from mmpose.apis import init_pose_model

def load_mmpose_model(cuda_device = "cpu"):
    '''
    Returns mmpose model that can be passed to inference_top_down_pose_model to infer pose keypoints
    '''
    model = init_pose_model(
        config="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py",
        checkpoint="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
        device=cuda_device
    )
    return model
