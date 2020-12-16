import io
import torch
import torch.onnx
import torch
import torchvision
from nnet.py_factory import NetworkFactory
from config import system_configs
import argparse
import os
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Demo CornerNet")
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
args = parse_args()
cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
print("cfg_file: {}".format(cfg_file))

with open(cfg_file, "r") as f:
    configs = json.load(f)

configs["system"]["snapshot_name"] = args.cfg_file
system_configs.update_config(configs["system"])


model = NetworkFactory(None)

# pthfile = 'cache/nnet/ExtremeNet/ExtremeNet_70000_64.pkl'
# loaded_model = torch.load(pthfile, map_location='cpu')
# try:
#   loaded_model.eval()
# except AttributeError as error:
#   print(error)

model.load_pretrained_params(args.model_path)
model.eval_mode()
# model = model.to(device)

# data type nchw
dummy_input1 = torch.randn(1, 3, 511, 511)
# dummy_input2 = torch.randn(1, 3, 64, 64)
# dummy_input3 = torch.randn(1, 3, 64, 64)
input_names = ["pre"]
output_names = ["r_heats"]
# torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(model.test([dummy_input1], debug=False), dummy_input1, "C3AE_emotion.onnx", verbose=True, input_names=None,
                  output_names=None)