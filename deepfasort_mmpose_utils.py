import torch
from mmpose_preprocess import preprocess_image
from mmpose.apis import init_pose_model
from mmpose.apis.inference import inference_top_down_pose_model
from mmpose.datasets import DatasetInfo

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

def single_image_preprocess(img_tensor:torch.Tensor, model=None):
    if model==None:
        model = load_mmpose_model()
    img_ndarray = img_tensor.numpy(force=True)
    result, _ = inference_top_down_pose_model(
        model,
        img_ndarray,
        dataset=model.cfg.data['test']['type'],
        dataset_info=DatasetInfo(model.cfg.data['test'].get('dataset_info', None)),
        return_heatmap=False,
        outputs=None
    )
    preprocessed_image = preprocess_image(img_ndarray, result)
    return torch.from_numpy(preprocessed_image)
