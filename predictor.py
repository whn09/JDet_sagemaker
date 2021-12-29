# -*- coding: utf-8 -*-
import json
import os
import warnings
import flask
import boto3
import io

import time
import jittor as jt
import numpy as np

import sys
sys.path.append('/opt/ml/code/JDet/python/')

# from jdet.runner import Runner 
from runner import Runner
from jdet.config import init_cfg

from PIL import Image
from jdet.data.transforms import Compose

# turn on cuda
jt.flags.use_cuda = 1

import yaml
config_file = '/opt/ml/model/config.yaml'
with open(config_file, "r") as f:
    cfg = yaml.load(f.read(), Loader=yaml.Loader)
print('cfg:', cfg)
# if 'dataset' in cfg:
#     del cfg['dataset']
with open(config_file,"w") as f:
    f.write(yaml.safe_dump(cfg, default_flow_style=False))

transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,),
        ]
transforms = Compose(transforms)

def get_data(filename):
    image = Image.open(filename).convert("RGB")
    target = dict(
        ori_img_size=image.size,
        img_size=image.size,
        scale_factor=1.,
        img_file = filename
    )

    image,target = transforms(image,target)
    return image,target


# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

init_cfg(config_file)
runner = Runner()
model = runner.model
model.eval()

print('init done.')

image, target = get_data('tmp.png')
inference_result = model(jt.array([image]), [target])
# print('inference_result:', inference_result)
# print(inference_result[0][0].shape, inference_result[0][1].shape, inference_result[0][2].shape)
boxes, scores, classes = inference_result[0]
result = {'boxes': boxes.numpy().tolist(), 'scores': scores.numpy().tolist(), 'classes': classes.numpy().tolist()}
_payload = json.dumps(result,ensure_ascii=False)
# print('_payload:', _payload)

print('test done')

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    # print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    #parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)

    if flask.request.content_type == 'application/x-image':
        image_as_bytes = io.BytesIO(flask.request.data)
        img = Image.open(image_as_bytes)
        download_file_name = '/tmp/tmp.png'
        img.save(download_file_name)
        print ("<<<<download_file_name ", download_file_name)
    else:
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)

        bucket = data['bucket']
        image_uri = data['image_uri']

        download_file_name = '/tmp/'+image_uri.split('/')[-1]
        print ("<<<<download_file_name ", download_file_name)

        try:
            s3_client.download_file(bucket, image_uri, download_file_name)
        except:
            #local test
            download_file_name = './tmp.png'

        print('Download finished!')

    image, target = get_data(download_file_name)
    inference_result = model(jt.array([image]), [target])
    boxes, scores, classes = inference_result[0]
    result = {'boxes': boxes.numpy().tolist(), 'scores': scores.numpy().tolist(), 'classes': classes.numpy().tolist()}
    _payload = json.dumps(result,ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')
