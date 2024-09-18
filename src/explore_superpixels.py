import torch
import os
import tarfile

def explore_mnist_superpixels_tar(file_path):
    print(f"Exploring file: {file_path}")
    
    with tarfile.open(file_path, 'r') as tar:
        print("Contents of the tar file:")
        tar.list()
        
        for member in tar.getmembers():
            if member.name.endswith('.pt'):
                print(f"\nExploring {member.name}")
                f = tar.extractfile(member)
                if f is not None:
                    data = torch.load(f)
                    explore_data(data, prefix="  ")

def explore_data(data, prefix=""):
    if isinstance(data, torch.Tensor):
        print(f"{prefix}Tensor:")
        print(f"{prefix}  Shape: {data.shape}")
        print(f"{prefix}  Dtype: {data.dtype}")
        print(f"{prefix}  Sample values: {data.flatten()[:5]}")
    elif isinstance(data, dict):
        print(f"{prefix}Dictionary:")
        for key, value in data.items():
            print(f"{prefix}  {key}:")
            explore_data(value, prefix + "    ")
    elif isinstance(data, list):
        print(f"{prefix}List of length {len(data)}")
        if data:
            print(f"{prefix}First element:")
            explore_data(data[0], prefix + "  ")
    elif isinstance(data, tuple):
        print(f"{prefix}Tuple of length {len(data)}")
        for i, item in enumerate(data):
            print(f"{prefix}Item {i}:")
            explore_data(item, prefix + "  ")
    else:
        print(f"{prefix}Type: {type(data)}")
        print(f"{prefix}Value: {data}")

# Path to the MNIST-superpixel.tar file
file_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'MNIST', 'raw', 'mnist_superpixels.tar.gz')

# Explore the file
explore_mnist_superpixels_tar(file_path)