# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Rekognition to
recognize people, objects, and text in images.

The usage demo in this file uses images in the .media folder. If you run this code
without cloning the GitHub repository, you must first download the image files from
    https://github.com/awsdocs/aws-doc-sdk-examples/tree/master/python/example_code/rekognition/.media
"""

# snippet-start:[python.example_code.rekognition.image_detection_imports]
import logging
from pprint import pprint
import boto3
from botocore.exceptions import ClientError

from rekognition_objects import (RekognitionLabel, show_bounding_boxes)

logger = logging.getLogger(__name__)

# snippet-end:[python.example_code.rekognition.image_detection_imports]


# snippet-start:[python.example_code.rekognition.RekognitionImage]
class RekognitionImage:
    """
    Encapsulates an Amazon Rekognition image. This class is a thin wrapper
    around parts of the Boto3 Amazon Rekognition API.
    """
    def __init__(self, image, image_name, rekognition_client):
        """
        Initializes the image object.

        :param image: Data that defines the image, either the image bytes or
                      an Amazon S3 bucket and object key.
        :param image_name: The name of the image.
        :param rekognition_client: A Boto3 Rekognition client.
        """
        self.image = image
        self.image_name = image_name
        self.rekognition_client = rekognition_client
# snippet-end:[python.example_code.rekognition.RekognitionImage]

# snippet-start:[python.example_code.rekognition.RekognitionImage.from_file]
    @classmethod
    def from_file(cls, image_file_name, rekognition_client, image_name=None):
        """
        Creates a RekognitionImage object from a local file.

        :param image_file_name: The file name of the image. The file is opened and its
                                bytes are read.
        :param rekognition_client: A Boto3 Rekognition client.
        :param image_name: The name of the image. If this is not specified, the
                           file name is used as the image name.
        :return: The RekognitionImage object, initialized with image bytes from the
                 file.
        """
        with open(image_file_name, 'rb') as img_file:
            image = {'Bytes': img_file.read()}
        name = image_file_name if image_name is None else image_name
        return cls(image, name, rekognition_client)
# snippet-end:[python.example_code.rekognition.RekognitionImage.from_file]

# snippet-start:[python.example_code.rekognition.RekognitionImage.from_bucket]
    @classmethod
    def from_bucket(cls, s3_object, rekognition_client):
        """
        Creates a RekognitionImage object from an Amazon S3 object.

        :param s3_object: An Amazon S3 object that identifies the image. The image
                          is not retrieved until needed for a later call.
        :param rekognition_client: A Boto3 Rekognition client.
        :return: The RekognitionImage object, initialized with Amazon S3 object data.
        """
        image = {'S3Object': {'Bucket': s3_object.bucket_name, 'Name': s3_object.key}}
        return cls(image, s3_object.key, rekognition_client)
# snippet-end:[python.example_code.rekognition.RekognitionImage.from_bucket]

    
# snippet-start:[python.example_code.rekognition.DetectLabels]
    def detect_labels(self, max_labels):
        """
        Detects labels in the image. Labels are objects and people.

        :param max_labels: The maximum number of labels to return.
        :return: The list of labels detected in the image.
        """
        try:
            response = self.rekognition_client.detect_labels(
                Image=self.image, MaxLabels=max_labels)
            labels = [RekognitionLabel(label) for label in response['Labels']]
            logger.info("Found %s labels in %s.", len(labels), self.image_name)
        except ClientError:
            logger.info("Couldn't detect labels in %s.", self.image_name)
            raise
        else:
            return labels
# snippet-end:[python.example_code.rekognition.DetectLabels]


# snippet-start:[python.example_code.rekognition.Usage_ImageDetection]
def usage_demo():
    print('-'*88)
    print("Welcome to the Amazon Rekognition image detection demo!")
    print('-'*88)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    rekognition_client = boto3.client('rekognition')
    street_scene_file_name = "picture.jpg"

    street_scene_image = RekognitionImage.from_file(
        street_scene_file_name, rekognition_client)
    input("Press Enter to continue.")
    print(f"Detecting labels in {street_scene_image.image_name}...")
    labels = street_scene_image.detect_labels(2)
    print(f"Found {len(labels)} labels.")
    for label in labels:
        pprint(label.to_dict())
    names = []
    box_sets = []
    colors = ['aqua', 'red', 'white', 'blue', 'yellow', 'green']
    for label in labels:
        if label.instances:
            names.append(label.name)
            box_sets.append([inst['BoundingBox'] for inst in label.instances])
    print(f"Showing bounding boxes for {names} in {colors[:len(names)]}.")
    show_bounding_boxes(
        street_scene_image.image['Bytes'], box_sets, colors[:len(names)])

    print("Thanks for watching!")
    print('-'*88)
# snippet-end:[python.example_code.rekognition.Usage_ImageDetection]

def recognize_img(path: str, labelsNum=5):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    rekognition_client = boto3.client('rekognition')
    img = RekognitionImage.from_file(path, rekognition_client)
    labels = img.detect_labels(labelsNum)
    labelsDict = {}
    for label in labels:
        if label.instances:
            labelsDict[label.name] = label.to_dict()
    return labelsDict

if __name__ == '__main__':
    usage_demo()
