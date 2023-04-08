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

## Model Visualizations :art:
Segmentation performance of each model is shown below:

![perf](https://github.com/eplatero97/LungSegmentationPerf/blob/master/assets/model_perf.png)

The first row represents the X-ray image of five lungs, second row represents the mask, and the rest are the generations of UNet, SegFormer, DeepLabV3+, PSPNet, and FCN respectively. 

## Dataset :file_folder:
To download the MOT17 dataset, execute below:
```bash
wget https://motchallenge.net/data/MOT17.zip -P data/
unzip data/MOT17.zip data/
```
To download the dataset, click [here](https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset/notebook#Lung-segmentation-from-Chest-X-Ray-dataset). Then, you must pre-process the dataset to format it in a way acceptable to mmsegmentation framework and to partition the dataset into training, validation, and testing set:
```bash
python ./data_prep.py --inputpath './archive/Lung Segmentation/' --outputpath ./data/lungsementation
```

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
RESUME_FROM=https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth
python mmpose_tools/train.py $CONFIG --resume-from=$RESUME_FROM --no-validate
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
