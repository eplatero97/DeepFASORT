<div align="center">
  <img src="resources/mmtrack-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmtrack)](https://pypi.org/project/mmtrack/)
[![PyPI](https://img.shields.io/pypi/v/mmtrack)](https://pypi.org/project/mmtrack)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmtracking.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmtracking/workflows/build/badge.svg)](https://github.com/open-mmlab/mmtracking/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmtracking/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmtracking)
[![license](https://img.shields.io/github/license/open-mmlab/mmtracking.svg)](https://github.com/open-mmlab/mmtracking/blob/master/LICENSE)

[📘Documentation](https://mmtracking.readthedocs.io/) |
[🛠️Installation](https://mmtracking.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://mmtracking.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmtracking.readthedocs.io/en/latest/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmtracking/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction :wave:
This repository evaluates the difference in performance in running DeepFASORT in three ways:
* without any Feature Amplification
* with a pre-trained Feature Amplification
* with a self-learned Feature Amplification 

## Findings :mag:

A summary of our findings where we trained DeepSORT component on half of the MOT17 training partition and used the other half as the test partition:

| Model       	   | Acc         | Prc        | Rcll        | Fscore      | IoU        |
| ----------- 	   | ----------  | ---------- | ----------- | ----------- | ---------- |
| DeepSORT         | **97.64%**  | **96.42%** | **97.64%**  | **97.01%**  | **94.23%** |
| DeepFASORT_pre   | **97.64%**  | 95.75%     | **97.64%**  | 96.64%      | 93.54%     |
| DeepFASORT_self  | 97.42%      | 95.56%     | 97.42%      | 96.43%      | 93.16%     |

A through review of all our findings is found on `ChestXraySegmentationAblationStudy_lightversion.pdf`.


## Dataset :file_folder:
To train and validate our results, we use the MOT17 dataset. Since this method is exclusively focused on the embedding aspect of DeepSORT, using MOT17 has the added benefit that it comes with bounding box detections from three different detection model (dpm, frcnn, and sdp).

To self-train our mmpose model, there are many different conversions and splits that we must due to MOT17.

To help you keep track of all the conversions/splits, after following this sections, you should have the following directory structure:
```bash
data/MOT17/
└─ reid/ # $REID
	├── faimgs/ # $REID_FAIMGS
	├	├── MOT17-02-FRCNN_000002/
	├	├── MOT17-02-FRCNN_000003/
	├	├── . . .
	├── imgs/ # $REID_IMGS
	├	├── MOT17-02-FRCNN_000002/
	├	├── MOT17-02-FRCNN_000003/
	├	├── . . .
	├── meta/ 
	├	├── train.txt # $REID_TRAIN_TXT
	├	├── train_80_coco.json # $REID_TRAIN_COCO
	├	├── train_80_kp_coco.json # $REID_TRAIN_KP_COCO
	├	├── train_80_self_learning_kp_coco.json # $REID_TRAIN_SL_KP_COCO
	├	├── train_80.txt
	├	├── train_20.txt
└─ test/
	├── MOT17-01-DPM/
	├── MOT17-01-FRCNN/
	├── . . .
└─ train_split/ # $TRAIN_SPLIT
	├── half-train_cocoformat.json
	├── half-train_detections.pkl
	├── . . .
└─ train/
	├── MOT17-02-DPM/
	├── MOT17-02-FRCNN/
	├── . . .
```  

For the initial configuration, the commented env variables are defined below:
```bash
MOT17=./data/MOT17 # put path where data lives
TRAIN_SPLIT=./data/MOT17/train_split
REID=./data/MOT17/reid
REID_TRAIN=data/MOT17/reid/meta/train_80.txt
REID_TRAIN_COCO=data/MOT17/reid/meta/train_80_coco.json
REID_IMGS=data/MOT17/reid/imgs/
REID_FAIMGS=./data/MOT17/reid/faimgs
REID_TRAIN_KP_COCO=./data/MOT17/reid/meta/train_80_kp_coco.json
DEVICE=cuda
THRESHOLD=.5
```


To start, let's download the MOT17 dataset:
```bash
wget https://motchallenge.net/data/MOT17.zip -P data/
unzip data/MOT17.zip -d data/
```

Then, since the MOT17 test ground-truths are not publically available, we split the training set into upper-half (train) and lower-half (validation):
```bash
python ./tools/convert_datasets/mot/mot2coco.py -i $MOT17 -o $TRAIN_SPLIT --convert-det --split-train
```

Since we want to self-train the mmpose model with only the re-embedded images, we need to extract each bounding box in the MOT17 training set:
```bash
python tools/convert_datasets/mot/mot2reid.py -i $MOT17 -o $REID --val-split 0.2 --vis-threshold 0.3
```

Now, convert the ReID dataset to coco format (format required to run mmpose with existing mmpose features) to feature amplify each of them with mmpose:
```bash
python reid_to_coco.py --input-file=$REID_TRAIN --output-file=$REID_TRAIN_COCO --image-root=$REID_IMGS 
```

Now, preprocess all train images using your mmpose model to extract keypoint predictions and feature amplified cropped images:
```bash
python mmpose_preprocess.py --device=$DEVICE --img-root=$REID_IMGS --json-input=$REID_TRAIN_COCO --output-dir=$REID_FAIMGS --json-output=$REID_TRAIN_KP_COCO
```

Now, filter `$REID_TRAIN_KP_COCO` to only include images above or equal to a certain threshold:
```bash
REID_TRAIN_SL_KP_COCO=./data/reid/meta/train_80_self_learning_kp_coco.json
REID_FAIMGS=./data/reid/faimgs/
python filter_dataset.py --json-input=$REID_TRAIN_KP_COCO --img-root=$REID_IMGS --threshold=$THRESHOLD --json-output=$REID_TRAIN_SL_KP_COCO --img-output=$REID_FAIMGS
```

## Training
In this project, we evaluate the performane of four-models:
* DeepSORT
* DeepSORT with coco-pretrained mmpose
* DeepSORT with coco-pretrained-selflearned mmpose
* DeepSORT with pretrained-selflearned and fa-reid mmpose

To do this, we need to train the mmpose and the reid network. The files we need to do this are below:
```bash
DeepFASORT
└─ configs/
	├── body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_vdeepfasort.py
	├── reid/
	├	├── resnet50_b32x8_MOT17.py
	├	├── resnet50_b32x8_faMOT17.py
└─ work_dirs/ # dir contains weights to be produced
	├── hrnet_w48_coco_256x192_vdeepfasort.pth # self-learned mmpose weights
	├── resnet50_b32x8_faMOT17.pth # fa trained reid weights
	├── resnet50_b32x8_MOT17.pth # trained reid weights
	


	├		├── CARTA_DS2-2022-03-09-embeddings
```

```bash
# self-train mmpose model
CONFIG=./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_vdeepfasort.py
python mmpose_train.py --config=$CONFIG --work-dir ./work_dirs/

# train reid model
REID_CFG=configs/reid/resnet50_b32x8_MOT17.py
python ./tools/train.py $REID_CFG $GPUS --work-dir ./work_dirs/

# fa-train reid model
FAREID_CFG=configs/reid/resnet50_b32x8_faMOT17.py
python ./tools/train.py $FAREID_CFG $GPUS --work-dir ./work_dirs/
```

## Evaluation
Now that we have our dataset and weights, we will now evaluate our models on the MOT17 lower-half split of the MOT17 training dataset:
```bash
# evaluate DeepSORT


# evaluate DeepSORT with pretrained mmpose

# evaluate DeepSORT with pretrained-selflearned-mmpose

# evaluate DeepSORT with pretrained-selflearned-mmpose and fa-trained reid

```

 
You should have the following directory structure:


## Configs :memo:
The configs to train each of the models is below:

* `configs/fcn/fcn_r18b-d8_512x1024_20k_chestxray_binary.py`
* `configs/pspnet/pspnet_r18b-d8_512x1024_10_chestxray.py`
* `configs/deeplabv3plus/deeplabv3plus_r18b-d8_512x1024_10_chestxray.py`
* `configs/segformer/segformer_mit-b0_8x1_1024x1024_10_chestxray.py`
* `configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_chestxray.py`

> **NOTE**: you will have to configure each of the config files to your own machine since I had some serious memory limitations on my local computer. 

## Self-Learning
To perform self-learning, run below:
```bash
CONFIG=/media/erick/9C33-6BBD/Github/mmlab/mmtracking/DeepFASORT/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_vdeepfasort.py
python mmpose_tools/train.py $CONFIG --no-validate
```

## Run Experiments :running:
To run training and validation, run below:
```bash
CONFIG_FILE=configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_chestxray.py
bash ./tools/dist_train.sh $CONFIG_FILE 1
```

To run on testing partition, run below:
```bash
CHECKPOINT_FILE=checkpoints/latest.pth 
CONFIG_FILE=configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_chestxray.py
python ./tools/test.py $CONFIG_FILE $CHECKPOINT_FILE --eval mIoU mDice mFscore 
```
> **NOTE**: use `test.sh` if you want to use distributed testing.
