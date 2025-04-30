"""
Test S3 Connection for Forza Dashboard

This script tests the connection to the S3 bucket before running the main application.
"""

import boto3
import os
import json
import time
from botocore.exceptions import ClientError, NoCredentialsError

def test_s3_connection(bucket_name, region=None):
    """Test connection to an S3 bucket
    
    Args:
        bucket_name: Name of the S3 bucket
        region: AWS region (optional)
        
    Returns:
        dict: Test results
    """
    results = {
        'success': False,
        'credentials_found': False,
        'bucket_exists': False,
        'bucket_writable': False,
        'errors': [],
        'messages': []
    }
    
    # Step 1: Check for AWS credentials
    try:
        if region:
            s3 = boto3.client('s3', region_name=region)
        else:
            s3 = boto3.client('s3')
        
        # Test if credentials are valid by listing buckets
        response = s3.list_buckets()
        results['credentials_found'] = True
        results['messages'].append("AWS credentials found and validated")
        
        # Print available buckets
        bucket_list = [b['Name'] for b in response['Buckets']]
        results['messages'].append(f"Available buckets: {', '.join(bucket_list) if bucket_list else 'No buckets found'}")
        
    except NoCredentialsError:
        results['errors'].append("AWS credentials not found. Run 'aws configure' to set them up.")
        return results
    except Exception as e:
        results['errors'].append(f"Error initializing S3 client: {str(e)}")
        return results
    
    # Step 2: Check if bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
        results['bucket_exists'] = True
        results['messages'].append(f"Bucket '{bucket_name}' exists and is accessible")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            results['messages'].append(f"Bucket '{bucket_name}' does not exist. Will need to be created.")
            
            # Try to create the bucket
            try:
                if region == 'us-east-1':
                    s3.create_bucket(Bucket=bucket_name)
                else:
                    s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={
                            'LocationConstraint': region
                        }
                    )
                results['bucket_exists'] = True
                results['messages'].append(f"Successfully created bucket '{bucket_name}'")
            except Exception as create_error:
                results['errors'].append(f"Failed to create bucket: {str(create_error)}")
                return results
        elif error_code == '403':
            results['errors'].append(f"Access denied to bucket '{bucket_name}'. Check your permissions.")
            return results
        else:
            results['errors'].append(f"Error checking bucket: {str(e)}")
            return results
    
    # Step 3: Test write access to the bucket
    test_key = f"test_file_{int(time.time())}.json"
    test_data = {
        'timestamp': time.time(),
        'message': 'This is a test file to verify S3 write access',
        'bucket': bucket_name
    }
    
    try:
        # Convert dict to JSON string
        test_content = json.dumps(test_data)
        
        # Upload the test file
        s3.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content
        )
        
        results['bucket_writable'] = True
        results['messages'].append(f"Successfully wrote test file to bucket: {test_key}")
        
        # Try to delete the test file
        try:
            s3.delete_object(
                Bucket=bucket_name,
                Key=test_key
            )
            results['messages'].append(f"Successfully deleted test file: {test_key}")
        except Exception as delete_error:
            results['messages'].append(f"Warning: Could not delete test file: {str(delete_error)}")
    
    except Exception as write_error:
        results['errors'].append(f"Failed to write to bucket: {str(write_error)}")
        return results
    
    # If we got here, all tests passed
    results['success'] = True
    return results

def print_test_results(results):
    """Print test results in a formatted way"""
    print("\n=== S3 CONNECTION TEST RESULTS ===")
    
    # Print overall status
    if results['success']:
        print("\n✅ SUCCESS: All tests passed!")
    else:
        print("\n❌ FAILED: Some tests did not pass.")
    
    # Print detailed results
    print("\nDetailed Results:")
    print(f"  AWS Credentials: {'✅ Found' if results['credentials_found'] else '❌ Not found'}")
    print(f"  Bucket Exists:   {'✅ Yes' if results['bucket_exists'] else '❌ No'}")
    print(f"  Bucket Writable: {'✅ Yes' if results['bucket_writable'] else '❌ No'}")
    
    # Print messages
    if results['messages']:
        print("\nInformation:")
        for msg in results['messages']:
            print(f"  ℹ️ {msg}")
    
    # Print errors
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  ❌ {error}")
    
    # Print next steps
    print("\nNext Steps:")
    if results['success']:
        print("  1. ✅ Your S3 configuration is working correctly")
        print("  2. 🚀 Run your application with auto-upload enabled")
    else:
        if not results['credentials_found']:
            print("  1. Run 'aws configure' to set up your AWS credentials")
            print("  2. Make sure you have permissions to access S3")
        elif not results['bucket_exists']:
            print("  1. Check if your bucket name is globally unique")
            print("  2. Verify you have permissions to create buckets")
        elif not results['bucket_writable']:
            print("  1. Check your IAM permissions for S3 write access")
            print("  2. Ensure your bucket policies allow writes")
    
    print("\n=====================================\n")

if __name__ == "__main__":
    # Get bucket name from environment or prompt user
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        bucket_name = input("Enter S3 bucket name: ")
    
    # Get region from environment or prompt user
    region = os.environ.get('AWS_DEFAULT_REGION')
    if not region:
        region = input("Enter AWS region (leave blank for default): ")
        if not region:
            region = None
    
    print(f"\nTesting connection to bucket '{bucket_name}' in region '{region or 'default'}'...")
    
    # Run the test
    results = test_s3_connection(bucket_name, region)
    
    # Print results
    print_test_results(results)