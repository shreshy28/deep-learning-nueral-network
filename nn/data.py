from pathlib import Path
import gzip
import hashlib
import urllib.error
import struct
import urllib.request
import numpy as np

SOURCES = ("https://storage.googleapis.com/cvdf-datasets/mnist/",
           "https://raw.githubusercontent.com/fgnt/mnist/master/")
# Original MNIST distribution MD5s detect corrupt or incorrect mirror files;
# HTTPS still provides transport authentication.
HASHES = ("f68b3c2dcbeaaa9fbdd348bbdeb94873", "d53e105ee54ea40749a09fcbcd1e9432",
          "9fb629c4189551a2d022fa330f9573f3", "ec29112dd5afa0611ce80d1b7f02629c")

def mnist(directory="data/mnist"):
    """Official MNIST IDX mirror; validate headers and payload sizes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = []
    for expected, name in zip(HASHES, ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                 "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")):
        path = directory/(name+".gz")
        if not path.exists():
            temporary = path.with_suffix(".part")
            for base in SOURCES:
                try:
                    with urllib.request.urlopen(base+name+".gz", timeout=60) as response:
                        temporary.write_bytes(response.read())
                    if hashlib.md5(temporary.read_bytes()).hexdigest() != expected:
                        raise ValueError("MNIST checksum mismatch")
                    temporary.replace(path)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if base == SOURCES[-1]:
                        raise
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"MNIST checksum mismatch: {path}")
        with gzip.open(path, "rb") as stream:
            magic, count = struct.unpack(">II", stream.read(8))
            if "images" in name:
                rows, cols = struct.unpack(">II", stream.read(8))
                if magic != 2051 or (rows, cols) != (28, 28):
                    raise ValueError("Invalid MNIST image header")
                a = np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, 784).astype(np.float32)/255
            else:
                if magic != 2049:
                    raise ValueError("Invalid MNIST label header")
                a = np.frombuffer(stream.read(), dtype=np.uint8).copy()
                if len(a) != count or np.any(a > 9):
                    raise ValueError("Invalid MNIST labels")
            arrays.append(a)
    return tuple(arrays)

def synthetic(seed=42, count=2400):
    """Three interleaved spirals, requiring nonlinear decision boundaries."""
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 3
    radius = rng.uniform(.05, 1, count)
    angle = radius*4 + y*2*np.pi/3 + rng.normal(0, .12, count)
    x = np.column_stack([radius*np.sin(angle), radius*np.cos(angle)])
    order = rng.permutation(count)
    return x[order], y[order]
