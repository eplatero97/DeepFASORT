'''
Script to filter COCO format JSON files including confidence values
Outputs JSON file including only images where average skeleton confidence is above threshold
with confidence values replaced with keypoint visibility = 2
and optionally copies those images from the root to a new directory
'''

import argparse
import json
import numpy as np
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-input", help="COCO format JSON file (with confidence values) to filter")
    parser.add_argument("--img-root", help="COCO format directory with preprocessed images")
    parser.add_argument("--threshold", help="Minimum confidence value to filter dataset on", type=float)
    parser.add_argument("--json-output", help="Path of filtered JSON file output", default="data/MOT17_filtered/annotations/person_keypoints_train.json")
    parser.add_argument("--img-output", help="Directory to save filtered images in", default=None)
    return parser.parse_args()

def filter_dataset(json_input, threshold):
    with open(json_input) as f:
        data = json.load(f)
    filtered_data = dict()
    filtered_data["categories"] = data["categories"]
    filtered_data["images"] = []
    filtered_data["annotations"] = []
    print(f"Filtering {len(data['annotations'])} images...")
    for img_index in range(len(data["annotations"])):
        skeleton = data["annotations"][img_index]["keypoints"]
        keypoint_cvs = [skeleton[3*joint+2] for joint in range(17)]
        avg_skeleton_cv = np.average(keypoint_cvs)
        if avg_skeleton_cv >= threshold:
            for joint in range(17):
                data["annotations"][img_index]["keypoints"][3*joint+2] = 2
            filtered_data["images"].append(data["images"][img_index])
            filtered_data["annotations"].append(data["annotations"][img_index])
    return filtered_data

def save_dataset(filtered_data, img_root, json_output, img_output):
    os.makedirs(os.path.dirname(json_output), exist_ok=True)
    with open(json_output, "w") as f:
        json.dump(filtered_data, f)
    if (img_output != None):
        print(f"Copying {len(filtered_data['images'])} images to output directory...")
        for image in filtered_data["images"]:
            filename = image["file_name"]
            src_path = os.path.join(img_root, filename)
            dst_path = os.path.join(img_output, filename)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)

def main():
    args = parse_args()
    filtered_data = filter_dataset(args.json_input, args.threshold)
    save_dataset(filtered_data, args.img_root, args.json_output, args.img_output)
    

if __name__ == "__main__":
    main()
