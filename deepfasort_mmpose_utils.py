import torch
from mmpose_preprocess import preprocess_image
from mmpose.apis import init_pose_model
from mmpose.apis.inference import inference_top_down_pose_model
from mmpose.datasets import DatasetInfo
from typing import Union
import numpy as np 
import imageio

class FeatureAmplification:
    def __init__(self, device='cpu', mmpose_config=None, mmpose_checkpoint=None):
        if mmpose_config is None:
            mmpose_config = "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
        if mmpose_checkpoint is None:
            mmpose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"

        self.model = self.load_mmpose_model(device, mmpose_config, mmpose_checkpoint)



    def load_mmpose_model(self, device, config, checkpoint):
        '''
        Returns mmpose model that can be passed to inference_top_down_pose_model to infer pose keypoints
        '''
        model = init_pose_model(
            config=config,
            checkpoint=checkpoint,
            device=device
        )
        return model

    def single_image_preprocess(self, img:Union[str, np.ndarray]):
        model = self.model
        is_tensor = False
        is_cuda = False
        if isinstance(img, str):
            img = imageio.imread(img)
        if isinstance(img, torch.Tensor):
            #img.shape # (1, 3, W, H)
            is_tensor = True
            is_cuda = img.is_cuda
            img = img.detach().cpu().numpy().squeeze().transpose((1,2,0)) # (W, H, 3)
        result, _ = inference_top_down_pose_model(
            model,
            img,
            dataset=model.cfg.data['test']['type'],
            dataset_info=DatasetInfo(model.cfg.data['test'].get('dataset_info', None)),
            return_heatmap=False,
            outputs=None
        )
        #print(f"result: {result}")
        preprocessed_image: np.ndarray = preprocess_image(img, result) # shape: (W, H, 3)
        if is_tensor:
            preprocessed_image = torch.from_numpy(preprocessed_image).permute(2,0,1).unsqueeze(0) # (1, 3, W, H)
            if is_cuda:
                preprocessed_image = preprocessed_image.cuda()
        return preprocessed_image

if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import PILToTensor
    import matplotlib.pyplot as plt
    model = load_mmpose_model('cuda')
    print(model)
    print(next(model.parameters()).is_cuda)
    # import image
    img_path = 'data/MOT17/reid/imgs/MOT17-02-FRCNN_000002/000000.jpg'
    out = single_image_preprocess(img_path, model, as_torch=True)
    print(out.shape)
#    plt.imshow(out)
#    plt.savefig("./test.png")
