import os
import pytest
import requests
import pyarrow.parquet as pq
import pyarrow
from urllib.parse import urlparse
from downloader import Downloader
import pandas as pd

temp_urls = [
    "https://media2.bollywoodhungama.in/wp-content/uploads/2016/04/60504069-346x260.jpg",
    "https://www.anikama.co.il/content/images/thumbs/0005499_im-not-getting-any-younger-magnet_600.jpeg",
    "https://cdn.abclocal.go.com/images/otrc/2010/photos/135284_4617_ful.jpg"

]

# Defining 3 test case inputs
@pytest.fixture
def downloader():

    tmpdir = './data'
    # Create a sample Parquet file with URLs
    pq_file = os.path.join(tmpdir, 'sample.parquet')
    schema = pyarrow.schema([('URL', 'string')])
    table = pyarrow.Table.from_pandas(pd.DataFrame({'URL': temp_urls}), schema=schema)
    pq.write_table(table, pq_file)
    
    # Instantiate the Downloader class with the sample Parquet file
    return Downloader(pq_file, str(tmpdir))

def test_download_image(downloader, tmpdir):
    # Call the download_image method for index 0
    image_path = downloader.download_image(0)
    
        # Set up test data

    # Verify that the image is downloaded and saved correctly
    assert os.path.isfile(image_path)
    
    # Check that the downloaded image content matches the expected content
    with open(image_path, 'rb') as f:
        assert f.read() == requests.get(temp_urls[0]).content

def test_getitem(downloader, tmpdir):
    # Call __getitem__ with a slice
    image_paths = downloader[0:2]
    
    # Verify that image_paths is a list containing two image paths
    assert isinstance(image_paths, list)
    assert len(image_paths) == 2
    
    # Verify that both images are downloaded and saved correctly
    for i, image_path in enumerate(image_paths):
        assert os.path.isfile(image_path)
        expected_url = temp_urls[i]
        with open(image_path, 'rb') as f:
            assert f.read() == requests.get(expected_url).content
