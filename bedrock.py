# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Bedrock to manage
Bedrock models.
"""

import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# snippet-start:[python.example_code.bedrock.BedrockWrapper.class]
# snippet-start:[python.example_code.bedrock.BedrockWrapper.decl]
class BedrockWrapper:
    """Encapsulates Amazon Bedrock foundation model actions."""

    def __init__(self, bedrock_client):
        """
        :param bedrock_client: A Boto3 Amazon Bedrock client, which is a low-level client that
                               represents Amazon Bedrock and describes the API operations for
                               creating and managing Bedrock models.
        """
        self.bedrock_client = bedrock_client

    # snippet-end:[python.example_code.bedrock.BedrockWrapper.decl]

    # snippet-start:[python.example_code.bedrock.GetFoundationModel]
    def get_foundation_model(self, model_identifier):
        """
        Get details about an Amazon Bedrock foundation model.

        :return: The foundation model's details.
        """

        try:
            return self.bedrock_client.get_foundation_model(
                modelIdentifier=model_identifier
            )["modelDetails"]
        except ClientError:
            logger.error(
                f"Couldn't get foundation models details for {model_identifier}"
            )
            raise

    # snippet-end:[python.example_code.bedrock.GetFoundationModel]

    # snippet-start:[python.example_code.bedrock.ListFoundationModels]
    def list_foundation_models(self):
        """
        List the available Amazon Bedrock foundation models.

        :return: The list of available bedrock foundation models.
        """

        try:
            response = self.bedrock_client.list_foundation_models()
            models = response["modelSummaries"]
            logger.info("Got %s foundation models.", len(models))
            return models

        except ClientError:
            logger.error("Couldn't list foundation models.")
            raise

    # snippet-end:[python.example_code.bedrock.ListFoundationModels]


# snippet-end:[python.example_code.bedrock.BedrockWrapper.class]