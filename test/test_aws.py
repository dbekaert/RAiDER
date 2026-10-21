from pathlib import Path

import pytest

from RAiDER import aws


@pytest.fixture
def s3_client(mocker):
    # Replace the module-level boto3 client so no test talks to real S3
    return mocker.patch.object(aws, 'S3_CLIENT')


@pytest.mark.parametrize('file_name', ['browse.png', 'dir/browse.png'])
def test_get_tag_set_browse(file_name):
    assert aws.get_tag_set(file_name) == {'TagSet': [{'Key': 'file_type', 'Value': 'browse'}]}


@pytest.mark.parametrize('file_name', ['product.nc', 'metadata.json', 'no_extension', 'browse.png.nc'])
def test_get_tag_set_product(file_name):
    assert aws.get_tag_set(file_name) == {'TagSet': [{'Key': 'file_type', 'Value': 'product'}]}


@pytest.mark.parametrize(
    'file_location, expected',
    [
        ('metadata.json', 'application/json'),
        ('browse.png', 'image/png'),
        (Path('browse.png'), 'image/png'),
        ('readme.txt', 'text/plain'),
    ],
)
def test_get_content_type_known(file_location, expected):
    assert aws.get_content_type(file_location) == expected


@pytest.mark.parametrize('file_location', ['no_extension', 'file.unknownext', Path('file.unknownext')])
def test_get_content_type_unknown(file_location):
    assert aws.get_content_type(file_location) == 'application/octet-stream'


def test_upload_file_to_s3_no_prefix(s3_client):
    path_to_file = Path('some/local/dir/browse.png')

    aws.upload_file_to_s3(path_to_file, 'my-bucket')

    s3_client.upload_file.assert_called_once_with(
        str(path_to_file), 'my-bucket', 'browse.png', {'ContentType': 'image/png'}
    )
    s3_client.put_object_tagging.assert_called_once_with(
        Bucket='my-bucket',
        Key='browse.png',
        Tagging={'TagSet': [{'Key': 'file_type', 'Value': 'browse'}]},
    )


def test_upload_file_to_s3_with_prefix(s3_client):
    path_to_file = Path('product.json')
    expected_key = str(Path('my-prefix') / 'product.json')

    aws.upload_file_to_s3(path_to_file, 'my-bucket', 'my-prefix')

    s3_client.upload_file.assert_called_once_with(
        'product.json', 'my-bucket', expected_key, {'ContentType': 'application/json'}
    )
    s3_client.put_object_tagging.assert_called_once_with(
        Bucket='my-bucket',
        Key=expected_key,
        Tagging={'TagSet': [{'Key': 'file_type', 'Value': 'product'}]},
    )


def test_upload_file_to_s3_logs(s3_client, mocker):
    logger = mocker.patch.object(aws, 'logger')

    aws.upload_file_to_s3(Path('product.nc'), 'my-bucket')

    logger.info.assert_called_once_with('Uploading s3://my-bucket/product.nc')


@pytest.fixture
def bucket_contents():
    return {
        'Contents': [
            {'Key': 'my-prefix/product.nc'},
            {'Key': 'my-prefix/metadata.json'},
            {'Key': 'my-prefix/browse.png'},
        ]
    }


@pytest.mark.parametrize(
    'file_type, expected_key',
    [
        ('.nc', 'my-prefix/product.nc'),
        ('.json', 'my-prefix/metadata.json'),
        ('.png', 'my-prefix/browse.png'),
    ],
)
def test_get_s3_file_found(s3_client, bucket_contents, file_type, expected_key):
    s3_client.list_objects_v2.return_value = bucket_contents

    result = aws.get_s3_file('my-bucket', 'my-prefix', file_type)

    s3_client.list_objects_v2.assert_called_once_with(Bucket='my-bucket', Prefix='my-prefix')
    s3_client.download_file.assert_called_once_with('my-bucket', expected_key, Path(expected_key).name)
    assert result == Path(Path(expected_key).name)


def test_get_s3_file_returns_first_match(s3_client):
    s3_client.list_objects_v2.return_value = {
        'Contents': [
            {'Key': 'my-prefix/first.nc'},
            {'Key': 'my-prefix/second.nc'},
        ]
    }

    result = aws.get_s3_file('my-bucket', 'my-prefix', '.nc')

    s3_client.download_file.assert_called_once_with('my-bucket', 'my-prefix/first.nc', 'first.nc')
    assert result == Path('first.nc')


def test_get_s3_file_no_match(s3_client, bucket_contents):
    s3_client.list_objects_v2.return_value = bucket_contents

    result = aws.get_s3_file('my-bucket', 'my-prefix', '.tif')

    s3_client.download_file.assert_not_called()
    assert result is None
