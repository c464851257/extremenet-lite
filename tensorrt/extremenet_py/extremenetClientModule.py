# -*- coding: utf-8 -*-
import json
import traceback
import torch

from external.nms import soft_nms_with_points as soft_nms
import cv2
from utils import crop_image, normalize_
import numpy as np
from PIL import Image
from loguru import logger
from functools import partial
from tensorrtserver.api import *
import tensorrtserver.api.model_config_pb2 as model_config
# import tritonclient.http as httpclient
from utils.visualize import vis_mask, vis_octagon, vis_ex, vis_class, vis_bbox
from kp_utils import _exct_decode
# import tritonclient.grpc as grpcclient


# in order to solve the linux Chinese coding problem
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

SHOW = False

####################################################################################################################################
mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
top_bboxes = {}
image_id = 0
categories = 1
nms_threshold = 0.5
nms_algorithm = {
    "nms": 0,
    "linear_soft_nms": 1,
    "exp_soft_nms": 2
}["exp_soft_nms"]
max_per_image = 100
suppres_ghost = True
class_name = [0,1,2,3,4,5]
image_ext = ['jpg', 'jpeg', 'png', 'webp','tif']
count = 0


@logger.catch
class extremenetRemoteInference(object):
    def __init__(self,protocol="grpc", url="localhost:8001",
        model_name="extremenet", model_version=1, batch_size=8, verbose=0, roi=None, model_args=None):
        self.roi = roi
        self.model_args = model_args
        self.num_classes = 80
        self.model_name = model_name
        self.img_list = []
        self.protocol = ProtocolType.from_str(protocol)
        self.input_name, self.output_name, \
        self.dtype, self.max_batch_size = self.parse_model(
            url, self.protocol, self.model_name, batch_size, verbose)
        logger.info("algorithm instance url is: {}".format(url))
        logger.info("model args is: {}".format(str(self.model_args)))
        logger.info("roi is: {}".format(str(self.roi)))
        # logger.info("model input names:{}".format(self.input_name))
        # logger.info("model output names:{}".format(self.output_name))
        # logger.info("model input image height:{}, width:{}, channel:{}, format:{}, dtype:{}".format(
        #    self.h, self.w, self.c, model_config.ModelInput.Format.Name(self.format), self.dtype))

        self.ctx = InferContext(url, self.protocol, self.model_name,
                       model_version, verbose, 0, False)
        # self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)
    def parse_model(self, url, protocol, model_name, batch_size, verbose=False):
        

        print("url:",url)
        print("protocol:",protocol)
        print("model_name:",model_name)
        print("verbose:",verbose)
        ctx = ServerStatusContext(url, protocol, model_name, verbose)
        server_status = ctx.get_server_status()
        if model_name not in server_status.model_status:
            raise Exception("unable to get status for '" + model_name + "'")

        status = server_status.model_status[model_name]
        config = status.config

        print("input:", config.input)
        print("output:", config.output)
        input_name = [x.name for x in config.input]
        output_name = [x.name for x in config.output]

        max_batch_size = config.max_batch_size

        # if max_batch_size == 0:
        #     if batch_size != 1:
        #         raise Exception("batching not supported for model '" + model_name + "'")
        # else:
        #     if batch_size > max_batch_size:
        #         raise Exception("expecting batch size <= {} for model {}".format(max_batch_size, model_name))
        
        input = config.input[0]

        
        return (input_name, output_name, self.model_dtype_to_np(input), max_batch_size)
    
    def model_dtype_to_np(self, model_dtype):
        if model_dtype == model_config.TYPE_BOOL:
            return np.bool
        elif model_dtype == model_config.TYPE_INT8:
            return np.int8
        elif model_dtype == model_config.TYPE_INT16:
            return np.int16
        elif model_dtype == model_config.TYPE_INT32:
            return np.int32
        elif model_dtype == model_config.TYPE_INT64:
            return np.int64
        elif model_dtype == model_config.TYPE_UINT8:
            return np.uint8
        elif model_dtype == model_config.TYPE_UINT16:
            return np.uint16
        elif model_dtype == model_config.TYPE_FP16:
            return np.float16
        elif model_dtype == model_config.TYPE_FP32:
            return np.float32
        elif model_dtype == model_config.TYPE_FP64:
            return np.float64
        elif model_dtype == model_config.TYPE_STRING:
            return np.dtype(object)
        return None

    def _rescale_dets(self, detections, ratios, borders, sizes):
        xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
        xs /= ratios[:, 1][:, None, None]
        ys /= ratios[:, 0][:, None, None]
        xs -= borders[:, 2][:, None, None]
        ys -= borders[:, 0][:, None, None]
        np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
        np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

    def _rescale_ex_pts(self, detections, ratios, borders, sizes):
        xs, ys = detections[..., 5:13:2], detections[..., 6:13:2]
        xs /= ratios[:, 1][:, None, None]
        ys /= ratios[:, 0][:, None, None]
        xs -= borders[:, 2][:, None, None]
        ys -= borders[:, 0][:, None, None]
        np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
        np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

    def _box_inside(self, box2, box1):
        inside = (box2[0] >= box1[0] and box2[1] >= box1[1] and \
                  box2[2] <= box1[2] and box2[3] <= box1[3])
        return inside

    
    def run(self,image):
        height, width = image.shape[0:2]

        detections = []
        scale = 1
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
        inputs = []
        outputs = []
        # inputs.append(grpcclient.InferInput('unscaledmels__0', [images.shape[0], images.shape[1],
        #                                                         images.shape[2], images.shape[3]], "FP32"))

        # Initialize the data
        inputs.append(images)
        # inputs[0].set_data_from_numpy(images)

        # outputs.append(grpcclient.InferRequestedOutput('audiodata__0'))
        # outputs.append(grpcclient.InferRequestedOutput('audiodata__1'))
        # outputs.append(grpcclient.InferRequestedOutput('audiodata__2'))
        # outputs.append(grpcclient.InferRequestedOutput('audiodata__3'))
        # outputs.append(grpcclient.InferRequestedOutput('audiodata__4'))
        # outputs.append(grpcclient.InferRequestedOutput('audiodata__5'))
        # outputs.append(grpcclient.InferRequestedOutput('audiodata__6'))
        # outputs.append(grpcclient.InferRequestedOutput('audiodata__7'))
        # outputs.append(grpcclient.InferRequestedOutput('audiodata__8'))
        # results = self.triton_client.infer(model_name=self.model_name,
        #                           inputs=inputs,
        #                           outputs=outputs,
        #                           client_timeout=None,
        #                           headers={'test': '1'})
        # 推理
        results_raw = self.ctx.run(
            {self.input_name[0]: inputs},
            {self.output_name[0]: (InferContext.ResultFormat.RAW),
             self.output_name[1]: (InferContext.ResultFormat.RAW),
             self.output_name[2]: (InferContext.ResultFormat.RAW),
             self.output_name[3]: (InferContext.ResultFormat.RAW),
             self.output_name[4]: (InferContext.ResultFormat.RAW),
             self.output_name[5]: (InferContext.ResultFormat.RAW),
             self.output_name[6]: (InferContext.ResultFormat.RAW),
             self.output_name[7]: (InferContext.ResultFormat.RAW),
             self.output_name[8]: (InferContext.ResultFormat.RAW)}, 1
        )

        t_heat = torch.from_numpy(results_raw['audiodata__0'][0])
        l_heat = torch.from_numpy(results_raw['audiodata__1'][0])
        b_heat = torch.from_numpy(results_raw['audiodata__2'][0])
        r_heat = torch.from_numpy(results_raw['audiodata__3'][0])
        ct_heat = torch.from_numpy(results_raw['audiodata__4'][0])
        t_regr = torch.from_numpy(results_raw['audiodata__5'][0])
        l_regr = torch.from_numpy(results_raw['audiodata__6'][0])
        b_regr = torch.from_numpy(results_raw['audiodata__7'][0])
        r_regr = torch.from_numpy(results_raw['audiodata__8'][0])

        # 后处理
        dets = _exct_decode(t_heat, l_heat, b_heat, r_heat, ct_heat, \
                            t_regr, l_regr, b_regr, r_regr)

        dets = dets.reshape(2, -1, 14).numpy()
        dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
        dets[1, :, [5, 7, 9, 11]] = out_width - dets[1, :, [5, 7, 9, 11]]
        dets[1, :, [7, 8, 11, 12]] = dets[1, :, [11, 12, 7, 8]].copy()

        self._rescale_dets(dets, ratios, borders, sizes)
        self._rescale_ex_pts(dets, ratios, borders, sizes)
        dets[:, :, 0:4] /= scale
        dets[:, :, 5:13] /= scale
        detections.append(dets)
        detections = np.concatenate(detections, axis=1)

        classes = detections[..., -1]
        classes = classes[0]
        detections = detections[0]

        keep_inds = (detections[:, 4] > 0)
        detections = detections[keep_inds]
        classes = classes[keep_inds]

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
            kth = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, 4] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]

        if suppres_ghost:
            for j in range(1, categories + 1):
                n = len(top_bboxes[image_id][j])
                for k in range(n):
                    inside_score = 0
                    if top_bboxes[image_id][j][k, 4] > 0.4:
                        for t in range(n):
                            if self._box_inside(top_bboxes[image_id][j][t],
                                           top_bboxes[image_id][j][k]):
                                inside_score += top_bboxes[image_id][j][t, 4]
                        if inside_score > top_bboxes[image_id][j][k, 4] * 3:
                            top_bboxes[image_id][j][k, 4] /= 2
        ex_point = {'box':[]}
        for j in range(1, categories + 1):
            keep_inds = (top_bboxes[image_id][j][:, 4] > 0.3)
            if len(top_bboxes[image_id][j][keep_inds]) == 0:
                cv2.imwrite('1.jpg', image)
                count += 1
            for bbox in top_bboxes[image_id][j][keep_inds]:
                sc = round(bbox[4], 2)
                ex = bbox[5:13].astype(np.int32).reshape(4, 2)
                is_vertical = (max(ex[0][0], ex[1][0], ex[2][0], ex[3][0])
                               - min(ex[0][0], ex[1][0], ex[2][0], ex[3][0])) < 30
                is_horizontal = (max(ex[0][1], ex[1][1], ex[2][1], ex[3][1])
                                 - min(ex[0][1], ex[1][1], ex[2][1], ex[3][1])) < 30
                is_ver_or_hor = is_vertical or is_horizontal
                image = vis_octagon(
                    image, ex, is_ver_or_hor)
                ex = ex.tolist()
                cv2.imwrite('1.jpg', image)
                self.img_list = ex
        # return ex_point

    def create_result_dictionary(self):
        rect = []
        result = [0]

        rect.append({'box':self.img_list,'status':0,'info':'mark'})
        
        return result,rect
####################################################################################################################################

# @logger.catch
def AlgorithmInitModle(GPUID, log_path, model_args=None, address="localhost", port="8001", roi=None):

    logger.add(log_path, rotation="1 MB", backtrace=False, level="DEBUG")
    logger.info("python: GPUID received: %d" % GPUID)
    logger.info("log_path received: %s\n" % log_path)

    #生成INFO.log
    logger.add(log_path,
                level="INFO",
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="00:00",
                retention="1 month")
    # # 生成ERROR.log
    logger.add(log_path,
               level="ERROR",
               format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
               rotation="00:00",
               retention="1 month")

    # 初始化模型远程推理类
    # global UnetInference
    try:
        url = address + ":" + port
        model_args = json.loads(model_args)
        extremenetInference = extremenetRemoteInference(url=url, roi=roi, model_args=model_args)
        return int(1), extremenetInference
    except Exception as e:
        # 捕获错误信息
        error_info = traceback.format_exc()
        logger.error(error_info)
        return int(0)


# @logger.catch
def AlgorithmGetProperty():

    GPUUseup = 0 # GPU ID
    alg_name = 'extremenet' # 算法名称
    logger.info("GPUUseup %f" %GPUUseup)
    logger.info("Algorithm name: %s" %alg_name)
    return True, GPUUseup, alg_name

# @logger.catch
def AlgorithmRun(cameraID, timeStamps, img, inference,dynamic_args=None):

    # global UnetInference
    # Yolov3Inference = Yolov3RemoteInference()  
    # init return values
    ret = False
    result = [0]
    boxes = []
    save_input = False
    save_output = False

    try:
        pic_size = list(img.shape)
        inference.run(img)
        result,rect = inference.create_result_dictionary()
        image = img.copy()
               
        detection_result = {"cameraID":cameraID,
                            "timeStamps": timeStamps,
                            "picSize":pic_size,
                            "boxes":rect,
                            "result":result,
                            "res1":"",
                            "res2":""
                                }
        ret = True

    except Exception as e:
        # 捕获错误信息
        error_info = traceback.format_exc()
        logger.error(error_info)
        image = img
        pic_size = list(image.shape)
        # 生成返回的json信息
        detection_result = {"cameraID":cameraID,
                            "timeStamps": timeStamps,
                            "picSize":pic_size,
                            "boxes":boxes,
                            "result":result,
                            "res1":"",
                            "res2":""
                            }

    #detectResult = json.dumps(detection_result)
    return ret, result, detection_result, save_input, save_output



