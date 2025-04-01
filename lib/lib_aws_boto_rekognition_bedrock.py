from rekognition import RekognitionImage
import boto3
import os
import json
import logging
from dotenv import load_dotenv, find_dotenv
import base64

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
aws_access_key_id = os.environ.get("aws_access_key_id")
aws_secret_access_key = os.environ.get("aws_secret_access_key")

def recognize_img(path: str, labelsNum=5):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    rekognition_client = boto3.client(
        'rekognition',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="eu-central-1"
    )
    img = RekognitionImage.from_file(path, rekognition_client)
    labels = img.detect_labels(labelsNum)
    labelsDict = {}
    for label in labels:
        if label.instances:
            labelsDict[label.name] = label.to_dict()
    return labelsDict

def generate_multimodal(image_path: str, prompt: str):
    boto3_bedrock = boto3.client(
        'bedrock-runtime',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="eu-central-1")
    image_base64 = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
    )

    modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    contentType = 'application/json'

    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    return response_body