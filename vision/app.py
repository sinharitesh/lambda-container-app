import json
import time
from io import BytesIO
import logging
import requests
import numpy as np
import os
import urllib
import boto3
s3_client = boto3.client('s3')

from fastai.vision.all import load_learner
logger = logging.getLogger()
logger.setLevel(logging.INFO)

mdl = load_learner('export.pkl')
FILE_BUCKET=os.environ.get('FILE_BUCKET')
TEMP_FILE_NAME = "/tmp/tempfile.png"

logger.info(f'File Bucket is {FILE_BUCKET}')

def download_file_from_s3_bucket(file_name):
    s3_client.download_file(FILE_BUCKET, file_name, TEMP_FILE_NAME)  


def download_file(url):
    file_name = '/tmp/test_file.png'
    urllib.request.urlretrieve(url, file_name)



def lambda_handler(event, context):
    """ Lambda function for prediction
    """
    logger.info(event)
    logger.info(f"Bucket Name: {FILE_BUCKET}")
    img_fname = ""
    try:
        logger.info(event['body'])
        request_body = event['body']
        if isinstance(request_body, str):
            request_body = json.loads(request_body)
            img_fname = request_body['fname']
    except:
        pass
    
    if img_fname == "":
        try:
            request_body = json.loads(event)
            img_fname = request_body['fname']
        except:
            pass
    
    logger.info(img_fname)
    start = time.time()
    fname = img_fname
    download_file_from_s3_bucket(file_name = fname)
    out = mdl.predict(TEMP_FILE_NAME)
    probability = -1
    try:
        probability = max(out[2].numpy())
    except:
        pass
    end = time.time()
    inference_time = np.round((end - start) * 1000, 2)
    logger.info(f'inference time for {img_fname}: {inference_time} ms')
    #message = f'For file:{img_fname}- result is: [{out[0]}] time taken: {str(inference_time)} ms'
    message = "API-OK"
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "prediciton": out[0],
                "probability": str(probability),
                "file_info": img_fname
            }
        ),
    }

