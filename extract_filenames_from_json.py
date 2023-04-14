import json
import argparse

parser = argparse.ArgumentParser(description= "Specify the filepath")

parser.add_argument('--input_file_1', default='data/MOT17/reid/coco/train_80.json')
parser.add_argument('--input_file_2', default='data/MOT17/half-train_cocoformat.json')
parser.add_argument('--input_file_3', default='data/MOT17/reid/meta/train_80.txt')
parser.add_argument('--output_file', default="data/MOT17/reid/train_upper-half.txt")


args = parser.parse_args()


input_file_1 = args.input_file_1
input_file_2 = args.input_file_2
input_file_3 = args.input_file_3
output_file = args.output_file
with open(input_file_1) as user_file:
    train_80 = json.load(user_file)

with open(input_file_2) as user_file:
    half_train_cocoformat = json.load(user_file)

train_80_ids = dict()
for i in range(len(train_80["images"])):
    train_80_ids[train_80["images"][i]["id"]] = train_80["images"][i]["file_name"]
    

half_train_cocoformat_ids = set()
for i in range(len(half_train_cocoformat["annotations"])):
    half_train_cocoformat_ids.add(half_train_cocoformat["annotations"][i]["image_id"])

d = {}
with open(input_file_3) as f:
    for line in f:
       (key, val) = line.split()
       d[key] = val
filenames = ""
for id in half_train_cocoformat_ids:
    filenames += 'data/MOT17/reid/imgs/' + train_80_ids[id] + " " +d[train_80_ids[id]] + "\n"

with open(output_file, "w") as text_file:
    text_file.write(filenames)
#print((filenames))