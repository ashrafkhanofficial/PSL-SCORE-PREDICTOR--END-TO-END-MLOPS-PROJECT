import boto3
import os
from src.utils.logger import get_logger
from src.exception import MyException
import sys
from dotenv import load_dotenv

load_dotenv()  
logger = get_logger(__name__)



class S3Client:
    def __init__(self):
        logger.info("Initializing S3 client...")
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_DEFAULT_REGION"),
            )
            logger.info("S3 client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize S3 client.")
            raise MyException(e, sys) from e

    def upload_file(self, file_path, bucket_name, object_name):
                try:
                    logger.info(f"Uploading file {file_path} to bucket {bucket_name} as {object_name}")

                    self.s3_client.upload_file(file_path, bucket_name, object_name)

                    logger.info("File uploaded successfully")

                except Exception as e:
                    logger.error("File upload to s3 bucket failed")
                    raise MyException(e, sys)


    def download_file(self, bucket_name, object_name, file_path):
        try:
            logger.info(f"Downloading {object_name} from bucket {bucket_name} to {file_path}")

            self.s3_client.download_file(bucket_name, object_name, file_path)

            logger.info("File downloaded successfully")

        except Exception as e:
            logger.error("File download from s3 bucket failed")
            raise MyException(e, sys)