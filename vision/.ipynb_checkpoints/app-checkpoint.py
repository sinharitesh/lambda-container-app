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
#import fastai
logger = logging.getLogger()
logger.setLevel(logging.INFO)

mdl = load_learner('export.pkl')
print("Learner loaded.")

def download_file_from_s3_bucket(bucket_name = "ecg-png-data", file_name = "s0558_re.png"):
    s3_client.download_file(bucket_name, file_name, "/tmp/s3.png")  
    return("done")


def download_file(url):
    file_name = '/tmp/test_file.png'
    urllib.request.urlretrieve(url, file_name)
#     mp3file = urllib2.urlopen(url)
#     with open('/tmp/test_file.png','wb') as output:
#       output.write(mp3file.read())
#     output.close


def lambda_handler(event, context):
    """Sample pure Lambda function
    """
    logger.info(event)
    print("Getting input object", event)
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
            print("SECOND TRY:", type(event))
            #if isinstance(event, dict):
            #request_body = json.loads(request_body)
            request_body = json.loads(event)
            img_fname = request_body['fname']
        except:
            pass
    
    logger.info(img_fname)
    print("IMGXXX:", img_fname)
    #input_object = input_fn(event['body'])
#     print("Type:", type(event))
#     fname = ""
#     if type(event) == dict:
#         fname = event['fname']
#     if type(event) == str:
#         jevent = json.loads(event)
#         fname =  jevent['fname']
        
    #print("FNAME:", jevent['fname'])
#     print(f"url is: {body['url']}")
    #fname = body['fname']
    # download_file(url) Works currently commenting out
    start = time.time()
    fname = img_fname
    download_file_from_s3_bucket(file_name = fname)
    #out = mdl.predict('/tmp/test_file.png')
    out = mdl.predict("/tmp/s3.png")
    print(out[1], out[2])
    probability = -1
    try:
        probability = max(out[2].numpy())
    except:
        pass
    end = time.time()
    #print(fname, out[0])
    inference_time = np.round((end - start) * 1000, 2)
    
    message = f'For file:{img_fname}- result is: [{out[0]}] time taken: {str(inference_time)} ms'
    #message = fastai.__version__ 
    #print("test")
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
    
    #---------------------------------------------------------------------------------------------------------#
    #print("Received event: " + json.dumps(event, indent=2))
#     body = json.loads(event['body'])
#     print(f"Body is: {body}")
#     url = body['url']
#     print(f"Getting image from URL: {url}")
#     response = requests.get(url)
#     print("Load image into memory")
#     img = PILImage.create(BytesIO(response.content))
#     print("Doing forward pass")
#     start = time.time()
#     pred,pred_idx,probs = learn.predict(img)
#     end = time.time()
#     inference_time = np.round((end - start) * 1000, 2)
#     print(f'class: {pred}, probability: {probs[pred_idx]:.04f}')
#     print(f'Inference time is: {str(inference_time)} ms')
#     return {
#         "statusCode": 200,
#         "body": json.dumps(
#             {
#                 "class": pred,
#                 "probability": "%.4f" % probs[pred_idx]
#             }
#         ),
#     }

