"""
Auto-Upload Module for Forza Dashboard

This module automatically uploads model analysis data to S3 when the application closes.
"""

import os
import atexit
import threading
import boto3
import json
import logging
import time
from botocore.exceptions import NoCredentialsError, ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='s3_auto_upload.log'
)
logger = logging.getLogger('s3_auto_upload')

class S3AutoUploader:
    """Class to handle automatic S3 uploads when the app closes"""
    
    def __init__(self, bucket_name, region=None, profile=None, source_dir="analysis_results"):
        """Initialize the auto uploader"""
        self.bucket_name = bucket_name
        self.source_dir = source_dir
        self.uploaded_files = []
        self.errors = []
        
        # Set up S3 client
        try:
            if profile:
                session = boto3.Session(profile_name=profile)
                self.s3 = session.client('s3', region_name=region)
            else:
                self.s3 = boto3.client('s3', region_name=region)
            
            # Test credentials by attempting a simple S3 operation
            # Just list buckets which requires minimal permissions
            self.s3.list_buckets()
            logger.info(f"S3 credentials validated for bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            self.s3 = None
    
    def ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, create it if it doesn't"""
        if not self.s3:
            logger.error("S3 client not initialized")
            return False
            
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists")
            return True
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                # Bucket doesn't exist, create it
                try:
                    # Create bucket with appropriate configuration based on region
                    if self.s3.meta.region_name == 'us-east-1':
                        # us-east-1 requires different syntax
                        self.s3.create_bucket(Bucket=self.bucket_name)
                    else:
                        # For all other regions, we need to specify LocationConstraint
                        self.s3.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={
                                'LocationConstraint': self.s3.meta.region_name
                            }
                        )
                    logger.info(f"Created bucket {self.bucket_name}")
                    return True
                except Exception as create_error:
                    logger.error(f"Failed to create bucket: {str(create_error)}")
                    return False
            else:
                logger.error(f"Error checking bucket: {str(e)}")
                return False
    
    def upload_file(self, file_path, s3_key=None):
        """Upload a single file to S3"""
        if not self.s3:
            logger.error("S3 client not initialized")
            return False
            
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # If no S3 key is provided, use the file name
        if s3_key is None:
            s3_key = os.path.basename(file_path)
        
        try:
            self.s3.upload_file(file_path, self.bucket_name, s3_key)
            self.uploaded_files.append(s3_key)
            logger.info(f"Uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            return False
        except Exception as e:
            logger.error(f"Error uploading {file_path}: {str(e)}")
            return False
    
    def upload_directory(self, directory, prefix=""):
        """Upload all files in a directory to S3"""
        if not self.s3:
            logger.error("S3 client not initialized")
            return False
            
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return False
        
        success = True
        file_count = 0
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Calculate the S3 key (path in the bucket)
                rel_path = os.path.relpath(file_path, directory)
                s3_key = os.path.join(prefix, rel_path).replace("\\", "/")
                
                # Upload the file
                if self.upload_file(file_path, s3_key):
                    file_count += 1
                else:
                    success = False
        
        logger.info(f"Uploaded {file_count} files from {directory}")
        return success
    
    def upload_all(self):
        """Upload all model analysis files to S3"""
        # Make sure the source directory exists
        if not os.path.exists(self.source_dir):
            logger.error(f"Source directory not found: {self.source_dir}")
            return False
        
        # Make sure the bucket exists
        if not self.ensure_bucket_exists():
            logger.error(f"Failed to ensure bucket exists: {self.bucket_name}")
            return False
        
        # Upload all files from the source directory
        success = self.upload_directory(self.source_dir)
        
        if success:
            logger.info(f"Successfully uploaded all files from {self.source_dir} to S3 bucket {self.bucket_name}")
        else:
            logger.error(f"Error uploading files to S3")
        
        return success

def register_s3_auto_upload(bucket_name, region=None, profile=None, source_dir="analysis_results", wait_secs=2):
    """Register a function to upload data to S3 when the app closes
    
    Args:
        bucket_name: Name of the S3 bucket
        region: AWS region (optional)
        profile: AWS profile name (optional)
        source_dir: Source directory to upload (default: "analysis_results")
        wait_secs: Seconds to wait before upload to ensure files are written (default: 2)
    """
    def upload_on_exit():
        """Function to run when the app exits"""
        # Sleep briefly to ensure all files are written
        logger.info(f"Application closing, waiting {wait_secs} seconds before upload...")
        time.sleep(wait_secs)
        
        # Initialize the uploader
        uploader = S3AutoUploader(bucket_name, region, profile, source_dir)
        
        # Upload the data
        try:
            uploader.upload_all()
            logger.info(f"Auto-upload complete. Uploaded {len(uploader.uploaded_files)} files to S3 bucket {bucket_name}")
        except Exception as e:
            logger.error(f"Error during auto-upload: {str(e)}")
    
    # Register the upload function to run when the app closes
    atexit.register(upload_on_exit)
    logger.info(f"Registered auto-upload to S3 bucket {bucket_name} on application exit")