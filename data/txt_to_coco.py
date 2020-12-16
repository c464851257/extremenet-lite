# -*- coding: UTF-8 -*-
import cv2
import json
import sys
import os
import numpy as np


# process bar
def process_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


root_path = "coco/images/train2017/"
val_images, categories, val_annotations = [], [], []
train_images, train_categories, train_annotations = [], [], []

category_dict = {"1": 1,"2": 2,"3": 3,"4": 4,"5": 5}

for cat_n in category_dict:
    categories.append({"supercategory": "", "id": category_dict[cat_n], "name": cat_n})

label_path = 'coco/images/labels/'
label_list = os.listdir(label_path)
number = 1
anno_id_count = 0
for label in label_list:
    img_id = int(label.split('/')[-1][:-4])
    label = os.path.join(label_path, label)
    with open(label, 'r') as f:
        count = 1
        total = 100
        for line in f.readlines():
            # process_bar(count, total)
            count += 1
            line = line.split(' ')
            img_name = str(img_id) + '.tif'
            bbox_num = int(line[1])
            img_cv2 = cv2.imread(root_path + img_name)
            [height, width, _] = img_cv2.shape

            # images info
            train_images.append({"file_name": img_name, "height": height, "width": width, "id": img_id})

            """
            annotation info:
            id : anno_id_count
            category_id : category_id
            bbox : bbox
            segmentation : [segment]
            area : area
            iscrowd : 0
            image_id : image_id
            """
            category_id = int(line[0])
            # for i in range(0, bbox_num):
            x1 = float(line[1])
            y1 = float(line[2])
            x2 = float(line[3])
            y2 = float(line[4])
            x3 = float(line[5])
            y3 = float(line[6])
            x4 = float(line[7])
            y4 = float(line[8])
            width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
            height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)

            bbox = [x1, y1, width, height]
            segment = [x1, y1, x2, y2, x3, y3, x4, y4]
            area = width * height

            anno_info = {'id': anno_id_count, 'category_id': category_id, 'bbox': bbox, 'segmentation': [segment],
                         'area': area, 'iscrowd': 0, 'image_id': img_id}
        # if number %10 != 0:
            train_annotations.append(anno_info)
            train_images.append({"file_name": img_name, "height": height, "width": width, "id": img_id})
            # else:
            #     val_annotations.append(anno_info)
            #     val_images.append({"file_name": img_name, "height": height, "width": width, "id": img_id})
            anno_id_count += 1

        f.close()

    number += 1

train_json = {"images": train_images, "annotations": train_annotations, "categories": categories}
val_json = {"images": val_images, "annotations": val_annotations, "categories": categories}

with open("train.json", "w") as outfile1:
    json.dump(train_json, outfile1)
with open("val.json", "w") as outfile2:
    json.dump(val_json, outfile2)