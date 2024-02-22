"""
Optimizing the Downloader class


"""

from pyarrow import parquet as pq
import os
import requests
from concurrent.futures import ThreadPoolExecutor


class Downloader:
    def __init__(self, pq_file, download_path="./data"):
        self.pq_file = pq_file
        self.table = pq.read_table(pq_file)
        self.links_list = self.table.column("URL").to_pylist()
        self.download_path = download_path
        if not os.path.exists(download_path):
            os.mkdir(download_path)

    def download_image(self, i):
        # Retrieve the link
        link = self.links_list[i]
        # Create the image path in the download dir
        image_path = os.path.join(self.download_path, f"image_{i}.jpg")
        if not os.path.exists(image_path):
            try:
                # Download image
                response = requests.get(link).content
                # Save image
                with open(image_path, "wb") as f:
                    f.write(response)
            except Exception as e:
                print(f"{i} not retrievable")
                return None
        return image_path

    def paralell_download_images(self, start, stop):
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.download_image, range(start, stop)))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.download_image(key)
        elif isinstance(key, slice):
            return self.paralell_download_images(key.start, key.stop)


if __name__ == "__main__":
    pq_file = "../task_4/links.parquet"
    d = Downloader(pq_file)
    print(d[10002:10010])
