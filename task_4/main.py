'''
Task 4

Download links.parquet
Download and save the first 10,000 images from the links in the parquet file.

- Monitor CPU and network usage during download.

References:
pyarrow package
requests package
'''
from multiprocessing import Pool
import os
import requests
from pyarrow import parquet as pq
from urllib.parse import urlparse



def download_image(self,i):
    # Retrieve the link
    link  = self.links_list[i]
    # Create the image path in the download dir
    image_path = os.path.join(self.download_path,f'image_{i}')
    if not os.path.exists(image_path):
        try:
            # Extract file extension from the URL
            parsed_url = urlparse(link)
            _, extension = os.path.splitext(parsed_url.path)
            # Download image
            response = requests.get(link)
            if extension:
                # Save image with appropriate extension
                with open(f'./data/image_{i}{extension}', 'wb') as f:
                    f.write(response.content)
            # If extension detection fails, save as .jpg
            with open(f'./data/image_{i}.jpg', 'wb') as f:
                f.write(response.content)
            image_path = os.path.join(self.download_path,f'image_{i}{extension}')
        except Exception as e:
            print("Error: ",e)
            print(f"{i} not retrievable")
            return None
    return image_path
            
def process_images(p_file, images_len):
    # Read the parquet file
    table = pq.read_table(p_file)
    # Get the links column
    links_list = table.column('URL').to_pylist()[:images_len]
    # Create a pool of workers
    p = Pool()
    # Create a mapping of the arguments
    args_list = [(link, i) for i, link in enumerate(links_list)]
    p.imap_unordered(download_image, args_list)
    p.close()  # Prevents any new tasks from being submitted
    p.join()   # Waits for all tasks to complete


if __name__ == "__main__":
    images_len = 10000
    # Path to parquet file
    parquet_path = './links.parquet'
    process_images(parquet_path,images_len)

