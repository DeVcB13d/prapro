import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from main import download_image, process_images
from multiprocessing import Pool
import random
import requests

temp_urls = [
    "https://media2.bollywoodhungama.in/wp-content/uploads/2016/04/60504069-346x260.jpg",
    "https://www.anikama.co.il/content/images/thumbs/0005499_im-not-getting-any-younger-magnet_600.jpeg",
    "https://cdn.abclocal.go.com/images/otrc/2010/photos/135284_4617_ful.jpg",
]
image_content = b"dummy image content"

img_dir = "./data"


# Defining some test cases to run on the download_image function
@pytest.mark.parametrize(
    "args, expected_extension",
    [
        (
            (
                "https://media2.bollywoodhungama.in/wp-content/uploads/2016/04/60504069-346x260.jpg",
                1,
                img_dir,
            ),
            ".jpg",
        ),
        (
            (
                "https://www.anikama.co.il/content/images/thumbs/0005499_im-not-getting-any-younger-magnet_600.jpeg",
                2,
                img_dir,
            ),
            ".jpeg",
        ),
        (
            (
                "https://cdn.abclocal.go.com/images/otrc/2010/photos/135284_4617_ful.jpg",
                3,
                img_dir,
            ),
            ".jpg",
        ),
    ],
)
def test_download_image(args, expected_extension):
    link, i, img_dir = args

    # Set up test data
    image_content = b"dummy image content"
    responses = {
        temp_urls[0]: image_content,
        temp_urls[1]: image_content,
        temp_urls[2]: image_content,
    }

    # Mock requests.get to return predefined responses
    def mock_get(url):
        print("responses", responses)
        return type("Response", (object,), {"content": responses[url]})

    requests.get = mock_get

    # Call the function
    image_path = download_image(args)
    print("img_path", image_path)
    # Verify the behavior
    expected_path = os.path.join(img_dir, f"image_{i}{expected_extension}")
    print("Expected", expected_path)
    assert os.path.isfile(expected_path)
    with open(expected_path, "rb") as f:
        assert f.read() == image_content
