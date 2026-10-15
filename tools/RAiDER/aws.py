"""Helpers for moving RAiDER products in and out of AWS S3."""

from __future__ import annotations

from mimetypes import guess_type
from pathlib import Path

import boto3

from RAiDER.logger import logger


S3_CLIENT = boto3.client('s3')


def get_tag_set(file_name: str) -> dict:
    """Build the S3 object tag set for a file.

    Args:
        file_name: Name of the file being uploaded.

    Returns:
        A tag set marking ``.png`` files as ``browse`` and all other files as ``product``.
    """
    if file_name.endswith('.png'):
        file_type = 'browse'
    else:
        file_type = 'product'

    tag_set = {'TagSet': [{'Key': 'file_type', 'Value': file_type}]}
    return tag_set


def get_content_type(file_location: Path | str) -> str:
    """Guess the MIME type of a file from its name.

    Args:
        file_location: Path or name of the file.

    Returns:
        The guessed MIME type, or ``application/octet-stream`` if it cannot be guessed.
    """
    content_type = guess_type(file_location)[0]
    if content_type is None:
        content_type = 'application/octet-stream'
    return content_type


def upload_file_to_s3(path_to_file: Path, bucket: str, prefix: str = '') -> None:
    """Upload a file to S3 and tag it by file type.

    Args:
        path_to_file: Local path of the file to upload.
        bucket: Name of the destination bucket.
        prefix: Key prefix to upload the file under.
    """
    key = str(Path(prefix) / path_to_file.name)
    extra_args = {'ContentType': get_content_type(key)}

    logger.info(f'Uploading s3://{bucket}/{key}')
    S3_CLIENT.upload_file(str(path_to_file), bucket, key, extra_args)

    tag_set = get_tag_set(path_to_file.name)

    S3_CLIENT.put_object_tagging(Bucket=bucket, Key=key, Tagging=tag_set)


def get_s3_file(bucket_name: str, bucket_prefix: str, file_type: str) -> Path | None:
    """Download the first file under an S3 prefix that ends with a given suffix.

    The file is saved to the current working directory under its own name.

    Args:
        bucket_name: Name of the bucket to search.
        bucket_prefix: Key prefix to search under.
        file_type: Suffix to match, such as ``.nc``.

    Returns:
        Path to the downloaded file, or None if no key matches.
    """
    result = S3_CLIENT.list_objects_v2(Bucket=bucket_name, Prefix=bucket_prefix)
    for s3_object in result['Contents']:
        key = s3_object['Key']
        if key.endswith(file_type):
            file_name = Path(key).name
            logger.info(f'Downloading s3://{bucket_name}/{key} to {file_name}')
            S3_CLIENT.download_file(bucket_name, key, file_name)
            return Path(file_name)
