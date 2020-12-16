#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import importlib
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")

from config import system_configs
from nnet.py_factory import NetworkFactory

from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms_with_points as soft_nms
from utils.color_map import colormap
from utils.visualize import vis_mask, vis_octagon, vis_ex, vis_class, vis_bbox
from dextr import Dextr

torch.backends.cudnn.benchmark = False
import time

class_name = [
    0,1,2,3,4,5]

image_ext = ['jpg', 'jpeg', 'png', 'webp','tif']

def parse_args():
    parser = argparse.ArgumentParser(description="Demo CornerNet", argument_default='ExtremeNet')
    parser.add_argument("--cfg_file", help="config file", 
                        default='ExtremeNet', type=str)
    parser.add_argument("--demo", help="demo image path or folders",
                        default="data/coco/images/test1/", type=str)
    parser.add_argument("--model_path",
                        default='cache/nnet/ExtremeNet/ExtremeNet_70000_64.pkl')
    parser.add_argument("--show_mask", action='store_true',
                        help="Run Deep extreme cut to obtain accurate mask")

    args = parser.parse_args()
    return args

def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

def _rescale_ex_pts(detections, ratios, borders, sizes):
    xs, ys = detections[..., 5:13:2], detections[..., 6:13:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

def _box_inside(box2, box1):
    inside = (box2[0] >= box1[0] and box2[1] >= box1[1] and \
       box2[2] <= box1[2] and box2[3] <= box1[3])
    return inside 

def kp_decode(nnet, images, K, kernel=3, aggr_weight=0.1,
              scores_thresh=0.1, center_thresh=0.1, debug=False):
    detections = nnet.test(
        [images], kernel=kernel, aggr_weight=aggr_weight, 
        scores_thresh=scores_thresh, center_thresh=center_thresh, debug=False)
    st = time.time()
    print(detections.device)
    detections = detections.data.cpu().numpy()
    print(detections.shape)
    print('detections:{}'.format(time.time() - st))
    return detections
# result = open('result.txt', 'w')
if __name__ == "__main__":
    args = parse_args()
    cfg_file = os.path.join(
        system_configs.config_dir, args.cfg_file + ".json")
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])
    print("system config...")
    pprint.pprint(system_configs.full)
    
    print("loading parameters: {}".format(args.model_path))
    print("building neural network...")
    nnet = NetworkFactory(None)
    print("loading parameters...")
    nnet.load_pretrained_params(args.model_path)
    # nnet.cuda()
    # device = 'gpu'
    # model = nnet.to(device)
    nnet.eval_mode()

    K             = configs["db"]["top_k"]
    aggr_weight   = configs["db"]["aggr_weight"]
    scores_thresh = configs["db"]["scores_thresh"]
    center_thresh = configs["db"]["center_thresh"]
    suppres_ghost = True
    nms_kernel    = 3
    
    scales        = configs["db"]["test_scales"]
    weight_exp    = 8
    categories    = configs["db"]["categories"]
    nms_threshold = configs["db"]["nms_threshold"]
    max_per_image = configs["db"]["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }["exp_soft_nms"]
    args.show_mask = False
    if args.show_mask:
        dextr = Dextr()

    # print(categories)
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
    std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
    top_bboxes = {}

    if os.path.isdir(args.demo):
        image_names = []
        ls = os.listdir(args.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(args.demo, file_name))
    else:
        image_names = [args.demo]

    for image_id, image_name in enumerate(image_names):
        print('Running ', image_name)
        image      = cv2.imread(image_name)

        start_time = time.time()
        height, width = image.shape[0:2]

        detections = []

        for scale in scales:
            new_height = int(height * scale)
            new_width  = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            inp_height = new_height | 127
            inp_width  = new_width  | 127

            images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios  = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes   = np.zeros((1, 2), dtype=np.float32)

            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio  = out_width  / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(
                resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.
            normalize_(resized_image, mean, std)
            # print(1111111)
            images[0]  = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0]   = [int(height * scale), int(width * scale)]
            ratios[0]  = [height_ratio, width_ratio]
            # print(444444)
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            images = torch.from_numpy(images)
            dets   = kp_decode(
                nnet, images, K, aggr_weight=aggr_weight, 
                scores_thresh=scores_thresh, center_thresh=center_thresh,
                kernel=nms_kernel, debug=True)
            dets   = dets.reshape(2, -1, 14)
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            dets[1, :, [5, 7, 9, 11]] = out_width - dets[1, :, [5, 7, 9, 11]]
            dets[1, :, [7, 8, 11, 12]] = dets[1, :, [11, 12, 7, 8]].copy()

            _rescale_dets(dets, ratios, borders, sizes)
            _rescale_ex_pts(dets, ratios, borders, sizes)
            dets[:, :, 0:4] /= scale
            dets[:, :, 5:13] /= scale
            detections.append(dets)

        detections = np.concatenate(detections, axis=1)
        # print(detections)
        # print('-----------------------------')

        classes    = detections[..., -1]
        classes    = classes[0]
        detections = detections[0]
        # print(detections)

        # reject detections with negative scores
        # print(detections[:, 4].shape)
        keep_inds  = (detections[:, 4] > 0)
        # print(keep_inds)
        detections = detections[keep_inds]
        classes    = classes[keep_inds]
        # print(detections)
        # print(classes)

        top_bboxes[image_id] = {}
        for j in range(categories):
            keep_inds = (classes == j)
            top_bboxes[image_id][j + 1] = \
                detections[keep_inds].astype(np.float32)
            soft_nms(top_bboxes[image_id][j + 1], 
                     Nt=nms_threshold, method=nms_algorithm)

        scores = np.hstack([
            top_bboxes[image_id][j][:, 4] 
            for j in range(1, categories + 1)
        ])
        if len(scores) > max_per_image:
            kth    = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, 4] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]

        if suppres_ghost:
            for j in range(1, categories + 1):
                n = len(top_bboxes[image_id][j])
                for k in range(n):
                    inside_score = 0
                    if top_bboxes[image_id][j][k, 4] > 0.2:
                        for t in range(n):
                            if _box_inside(top_bboxes[image_id][j][t], 
                                           top_bboxes[image_id][j][k]):
                                inside_score += top_bboxes[image_id][j][t, 4]
                        if inside_score > top_bboxes[image_id][j][k, 4] * 3:
                            top_bboxes[image_id][j][k, 4] /= 2
        infer_time_end = time.time()
        infer_time = infer_time_end - start_time

        if 1: # visualize
            color_list    = colormap(rgb=True)
            mask_color_id = 0
            image         = cv2.imread(image_name)
            input_image   = image.copy()
            mask_image    = image.copy()
            bboxes = {}
            # print(categories)
            for j in range(1, categories + 1):
                # print(top_bboxes)
                # print(j)
                keep_inds = (top_bboxes[image_id][j][:, 4] > 0.5)
                cat_name  = class_name[j]
                # print(cat_name)
                # print(top_bboxes)
                for bbox in top_bboxes[image_id][j][keep_inds]:
                    # print('ssssss', cat_name)
                    sc    = round(bbox[4], 2)
                    # print('sc:{}'.format(sc))
                    ex    = bbox[5:13].astype(np.int32).reshape(4, 2)
                    # print('ex:{}'.format(ex))
                    bbox  = bbox[0:4].astype(np.int32)
                    # print('bbox:{}'.format(bbox))
                    txt   = '{}{:.2f}'.format(cat_name, sc)
                    # print('txt:{}'.format(txt))
                    # print(image_name.split('/')[-1])
                    # result.write(str(image_name.split('/')[-1]))
                    # result.write(' ' + str(cat_name))
                    # result.write(' ' + str(sc))
                    # result.write(' ' + str(ex[0][0]))
                    # result.write(' ' + str(ex[0][1]))
                    # result.write(' ' + str(ex[1][0]))
                    # result.write(' ' + str(ex[1][1]))
                    # result.write(' ' + str(ex[2][0]))
                    # result.write(' ' + str(ex[2][1]))
                    # result.write(' ' + str(ex[3][0]))
                    # result.write(' ' + str(ex[0][1]))
                    # result.write('\n')

                    color_mask = color_list[mask_color_id % len(color_list), :3]
                    mask_color_id += 1
                    # image = vis_bbox(image,
                    #                  (bbox[0], bbox[1],
                    #                   bbox[2] - bbox[0], bbox[3] - bbox[1]))
                    # image = vis_class(image,
                    #                   (bbox[0], bbox[1] - 2), txt)
                    ex_time_end = time.time()
                    ex_time = ex_time_end - start_time
                    # is_vertical = (max(ex[0][0], ex[1][0], ex[2][0], ex[3][0])
                    #                     - min(ex[0][0], ex[1][0], ex[2][0], ex[3][0])) < 30
                    # is_horizontal = (max(ex[0][1], ex[1][1], ex[2][1], ex[3][1])
                    #                     - min(ex[0][1], ex[1][1], ex[2][1], ex[3][1])) < 30
                    # print(is_vertical, is_horizontal)
                    # if is_vertical or is_horizontal:
                    #     # point1 = (min(ex[0][0], ex[1][0], ex[2][0], ex[3][0]), min(ex[0][1], ex[1][1], ex[2][1], ex[3][1]))
                    #     # point2 = (max(ex[0][0], ex[1][0], ex[2][0], ex[3][0]), max(ex[0][1], ex[1][1], ex[2][1], ex[3][1]))
                    #     pts = np.array([[ex[0][0], ex[0][1]], [ex[1][0], ex[1][1]], [ex[2][0], ex[2][1]], [ex[3][0], ex[3][1]]])
                    #     # contours =
                    #     # cv2.rectangle(image, point1, point2, (0, 128, 0), 1)
                    #     cv2.drawContours(image, np.int32([pts]), -1, (0, 128, 0), 1, cv2.LINE_AA)
                    #     # cv2.polylines(image, np.int32([pts]), True, (0, 128, 0), 1)
                    # else:
                    #     print(max(ex[0][1], ex[1][1], ex[2][1], ex[3][1]), min(ex[0][1], ex[1][1], ex[2][1], ex[3][1]))
                    image = vis_octagon(
                        image, ex, color_mask)
                    # image = vis_ex(image, ex, color_mask)

                    if args.show_mask:
                        mask = dextr.segment(input_image[:, :, ::-1], ex) # BGR to RGB
                        mask = np.asfortranarray(mask.astype(np.uint8))
                        mask_image = vis_bbox(mask_image,
                                             (bbox[0], bbox[1],
                                              bbox[2] - bbox[0],
                                              bbox[3] - bbox[1]))
                        mask_image = vis_class(mask_image,
                                               (bbox[0], bbox[1] - 2), txt)
                        mask_image = vis_mask(mask_image, mask, color_mask)

            # if args.show_mask:
            #     cv2.imshow('mask', mask_image)
            #         print(os.path.join('out', image_name.split('/')[-1]))
                    print("infer_time: {}, get_ex_time: {}".format(infer_time, ex_time))
                    cv2.imwrite(os.path.join('out', image_name.split('/')[-1]), image)

            # cv2.waitKey()



