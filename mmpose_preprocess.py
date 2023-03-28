'''
Script to preprocess images using keypoints obtained from mmpose
Assumes input image directory is in COCO format
'''
import argparse
import cv2
import json
import mmcv
import os
import numpy as np
from xtcocotools.coco import COCO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model)
from mmpose.datasets import DatasetInfo

DEFAULTS = {
    "pose_config": "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py",
    "pose_checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
    "img-root": "tests/data/coco",
    "json-file": "tests/data/coco/test_coco.json",
    "device": "cpu",
    "dilation-kernel": 9,
    "output-dir": "mmpose_preprocessed_out",
    "json-output": "test_results.json",
    "return_heatmap": False,
    "output_layer_names": None
}
KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON =  [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_config', help='Config file for detection', default=DEFAULTS['pose_config'])
    parser.add_argument('--pose_checkpoint', help='Checkpoint file', default=DEFAULTS['pose_checkpoint'])
    parser.add_argument('--img-root', type=str, help='Image root', default=DEFAULTS['img-root'])
    parser.add_argument(
        '--json-file',
        type=str,
        default=DEFAULTS['json-file'],
        help='Json file containing image info.')
    parser.add_argument('--device', help='Device used for inference', default=DEFAULTS['device'],)
    parser.add_argument('--dilation-kernel', help='Size of kernel used for dilation of skeleton', default=DEFAULTS['dilation-kernel'])
    parser.add_argument("--output-dir", help='Output directory for preprocessed images', default=DEFAULTS['output-dir'])
    parser.add_argument("--json-output", help='Output annotation json file path', default=DEFAULTS['json-output'])
    return parser.parse_args()

def load_inputs(img_root, json_file):
    image_names = []
    all_person_results = []
    coco = COCO(json_file)
    img_keys = list(coco.imgs.keys())
    for i in range(len(img_keys)):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(img_root, image['file_name'])
        image_names.append(image_name)
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)
        all_person_results.append(person_results)

    return image_names, all_person_results

def mmpose_inferences(image_names, all_person_results, pose_model):
    print(f"Making inferences on {len(image_names)} images...")
    results = []
    for i in mmcv.track_iter_progress(range(len(image_names))):
        # test a single image, with a list of bboxes
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            image_names[i],
            all_person_results[i],
            bbox_thr=None,
            format='xywh',
            dataset=pose_model.cfg.data['test']['type'],
            dataset_info=DatasetInfo(pose_model.cfg.data['test'].get('dataset_info', None)),
            return_heatmap=DEFAULTS['return_heatmap'],
            outputs=DEFAULTS['output_layer_names'])
        results.append(pose_results)
    return results

def preprocess_images(image_names, results, dilation_kernel_size, output_dir, img_root):
    print("Masking images...")
    for i in range(len(image_names)):
        img = cv2.imread(image_names[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h = img.shape[0]
        img_w = img.shape[1]

        mask = np.zeros(img.shape)

        for person in results[i]:
            keypoints = person['keypoints']
            for keypoint in keypoints:
                cv2.circle(mask, (int(keypoint[0]), int(keypoint[1])), 4, [255, 255, 255], -1)
            for link in SKELETON:
                pos1 = (int(keypoints[link[0], 0]), int(keypoints[link[0], 1]))
                pos2 = (int(keypoints[link[1], 0]), int(keypoints[link[1], 1]))
                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h):
                    # skip the link that should not be drawn
                    continue
                cv2.line(mask, pos1, pos2, [255, 255, 255], thickness=1)

        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask_grey = cv2.cvtColor(np.uint8(mask), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(mask_grey, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, [255,255,255], -1)
        mask_grey = cv2.cvtColor(np.uint8(mask), cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(mask_grey, 127, 255, 0)
        preprocessed_image = cv2.bitwise_and(img, img, mask=thresh)
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR)
        
        output_filename = os.path.join(output_dir, os.path.relpath(image_names[i], img_root))
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        cv2.imwrite(output_filename, preprocessed_image)

def write_json(json_input, results, json_output):
    json_file = open(json_input)
    coco_dict = json.load(json_file)
    
    coco_dict["categories"][0]["keypoints"] = KEYPOINTS
    coco_dict["categories"][0]["skeleton"] = SKELETON

    for p in range(len(results)):
        coco_dict["annotations"][p]["keypoints"] = [val.item() for val in list(results[p][0]["keypoints"].flatten())]

    with open(json_output, "w") as f:
        json.dump(coco_dict, f)

def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    image_names, all_person_results = load_inputs(args.img_root, args.json_file)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    results = mmpose_inferences(image_names, all_person_results, pose_model)
    preprocess_images(image_names, results, args.dilation_kernel, args.output_dir, args.img_root)
    write_json(args.json_file, results, args.json_output)

if __name__ == '__main__':
    main()
