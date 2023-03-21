import json
import cv2
import os
import argparse

#Process the input arguments
parser = argparse.ArgumentParser(description= "Specify the filepath of REID meta files as input and coco json file as output")

parser.add_argument('input_file')
parser.add_argument('output_file')

args = parser.parse_args()
if args.input_file == "":
    print("Specify the input file (.txt)")
    exit(0)
if args.output_file == "":
    print("Specify the output file (.json)")
    exit(0)

input_file = args.input_file
output_file = args.output_file

# Load REID dataset file
with open(input_file, 'r') as f:
    reid_data = f.readlines()

# Initialize COCO annotation dictionary
coco_data = {
    'images': [],
    'annotations': [],
    'categories': [{'id': 1, 'name': 'person'}]
}

# Loop over images in REID dataset
for i, line in enumerate(reid_data):
    image_info = line.split()
    image_filename =os.path.join(os.getcwd(), "reid","imgs" , image_info[0]) 
    #print(image_filename)
    person_id = int(image_info[1])
    im = cv2.imread(image_filename)
    height, width, _ = im.shape
    x = 0
    y = 0
    coco_data["images"].append({
        "id": i+1,
        "file_name": image_info[0],
        "width": width,
        "height": height
    })
    coco_data["annotations"].append({
        "id": i+1,
        "image_id": i+1,
        "file_name": image_info[0],
        "bbox": [x, y, width, height]
    })

#Write to json file
with open(output_file, "w") as f:
    json.dump(coco_data, f)
    
    