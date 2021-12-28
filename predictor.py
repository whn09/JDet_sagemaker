# -*- coding: utf-8 -*-
import sys
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

from jdet.runner import Runner 
from jdet.config import init_cfg

from PIL import Image
from jdet.data.transforms import Compose

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
    img = Image.open(filename).convert("RGB")
    targets = dict(
        ori_img_size=img.size,
        img_size=img.size,
        scale_factor=1.,
        img_file = filename
    )

    img,targets = transforms(img,targets)
    return img,targets 


# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

config_file = '/opt/ml/model/config.yaml'
init_cfg(config_file)
runner = Runner()
model = runner.model
model.eval()

print('init done.')


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
    inference_result = model(image, target)
    
    _payload = json.dumps(inference_result,ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')
