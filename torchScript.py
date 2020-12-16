import torch
import cv2
import numpy as np
import argparse
import os
from models.py_utils import exkp, CTLoss, _neg_loss, convolution, residual
import json
from nnet.py_factory import NetworkFactory

from config import system_configs
from utils import crop_image, normalize_

scales = [1]

def parse_args():
    parser = argparse.ArgumentParser(description="Demo CornerNet")
    parser.add_argument("--cfg_file", help="config file",
                        default='ExtremeNet', type=str)
    parser.add_argument("--demo", help="demo image path or folders",
                        default="data/coco/images/test1/", type=str)
    parser.add_argument("--model_path",
                        default='ExtremeNet_70000_64.pkl')
    parser.add_argument("--show_mask", action='store_true',
                        help="Run Deep extreme cut to obtain accurate mask")

    args = parser.parse_args()
    return args
args = parse_args()
cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
print("cfg_file: {}".format(cfg_file))

with open(cfg_file, "r") as f:
    configs = json.load(f)

configs["system"]["snapshot_name"] = args.cfg_file
system_configs.update_config(configs["system"])
import torch.nn as nn
def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)
# # An instance of your model.
def load_pretrained_params(model, pretrained_model):
    print("loading from {}".format(pretrained_model))
    device = torch.device('cpu')
    with open(pretrained_model, "rb") as f:
        params = torch.load(f, map_location=device)
        # params = torch.load(f)
        model.load_state_dict(params, strict=False)

model = NetworkFactory(None)
print("loading parameters...")
model.load_pretrained_params(args.model_path)
# nnet.cuda()
# device = 'gpu'
# model = nnet.to(device)
model.eval_mode()
mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
image_ext = ['jpg', 'jpeg', 'png', 'webp','tif']
# An example input you would normally provide to your model's forward() method.
# example = torch.rand(1, 3, 640, 320)
if os.path.isdir(args.demo):
    image_names = []
    ls = os.listdir(args.demo)
    for file_name in sorted(ls):
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in image_ext:
            image_names.append(os.path.join(args.demo, file_name))
else:
    image_names = [args.demo]
count = 0
for image_id, image_name in enumerate(image_names):
    print('Running ', image_name)
    image = cv2.imread(image_name)
    height, width = image.shape[0:2]

    detections = []

    for scale in scales:
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_center = np.array([new_height // 2, new_width // 2])

        inp_height = new_height | 127
        inp_width = new_width | 127

        images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
        ratios = np.zeros((1, 2), dtype=np.float32)
        borders = np.zeros((1, 4), dtype=np.float32)
        sizes = np.zeros((1, 2), dtype=np.float32)

        out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
        height_ratio = out_height / inp_height
        width_ratio = out_width / inp_width

        resized_image = cv2.resize(image, (new_width, new_height))
        resized_image, border, offset = crop_image(
            resized_image, new_center, [inp_height, inp_width])

        resized_image = resized_image / 255.
        normalize_(resized_image, mean, std)
        images[0] = resized_image.transpose((2, 0, 1))
        borders[0] = border
        sizes[0] = [int(height * scale), int(width * scale)]
        ratios[0] = [height_ratio, width_ratio]
        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        # images = torch.tensor(images)
        # print(images.dtype.type)


# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        traced_script_module = torch.jit.trace(model, images)
        traced_script_module.save("traced_extremenet_model.pt")
        print('Successfully translated model!!!')
        break
