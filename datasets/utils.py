from tqdm import tqdm
import numpy as np
import os
import sys
import requests
import gzip

def download_file(url, path):

    filename = os.path.basename(url)
    download_path = os.path.join(path, filename)

    filesize = int(requests.head(url).headers["Content-Length"])
    chunk_size = 1024

    with requests.get(url, stream=True) as r, open(download_path, "wb") as f, tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=filesize,
            file=sys.stdout,
            desc=filename
    ) as progress:
        for chunk in r.iter_content(chunk_size=chunk_size):
            datasize = f.write(chunk)
            progress.update(datasize)
        
    return download_path


def load_ubyte(filepath, num_samples, shape, header_size):
    f = gzip.open(filepath)
    f.read(header_size)
    buff = f.read(num_samples * int(np.array(shape).prod()))
    data = np.frombuffer(buff, dtype=np.uint8)
    return data.reshape(num_samples, *shape)
