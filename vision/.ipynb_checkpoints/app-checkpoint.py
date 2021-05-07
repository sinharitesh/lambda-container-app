import json
import time
from io import BytesIO
import logging
import requests
import numpy as np

import urllib
import boto3
s3_client = boto3.client('s3')

from fastai.vision.all import load_learner
logger = logging.getLogger()
logger.setLevel(logging.INFO)

mdl = load_learner('export.pkl')

def download_file_from_s3_bucket(bucket_name = "ecg-png-data", file_name = "s0558_re.png"):
    s3_client.download_file(bucket_name, file_name, "/tmp/s3.png")  


def download_file(url):
    file_name = '/tmp/test_file.png'
    urllib.request.urlretrieve(url, file_name)



def lambda_handler(event, context):
    """ Lambda function for prediction
    """
    logger.info(event)
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
    out = mdl.predict("/tmp/s3.png")
    probability = -1
    try:
        probability = max(out[2].numpy())
    except:
        pass
    end = time.time()
    inference_time = np.round((end - start) * 1000, 2)
    
    message = f'For file:{img_fname}- result is: [{out[0]}] time taken: {str(inference_time)} ms'
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "prediciton": out[0],
                "probability": str(probability),
                "file_info": img_fname,
                "time_taken": f' {str(inference_time)} ms',
                "message": message
            }
        ),
    }

