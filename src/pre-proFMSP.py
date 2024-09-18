# NOT CURRENTLY USED IN THE train_models.py, WORK IN PROGRESS FOR FUTURE USE

import numpy as np
from skimage.segmentation import slic
import networkx as nx
import torch
from torch_geometric.data import Data
import torch
from torchvision import datasets
from tqdm import tqdm
import os
import torch

def get_graph_from_image(image, desired_nodes=75):
    # Ensure the image is 2D (grayscale)
    if len(image.shape) > 2:
        image = image.squeeze()
    
    segments = slic(image, n_segments=desired_nodes, slic_zero=True)
    asegments = np.array(segments)
    unique_segments = np.unique(asegments)
    num_nodes = len(unique_segments)

    nodes = {
        node: {
            "intensity_list": [],
            "pos_list": []
        } for node in unique_segments
    }

    height, width = image.shape

    for y in range(height):
        for x in range(width):
            node = asegments[y, x]
            intensity = image[y, x]
            pos = np.array([float(x)/width, float(y)/height])
            nodes[node]["intensity_list"].append(intensity)
            nodes[node]["pos_list"].append(pos)

    G = nx.Graph()

    for node in nodes:
        nodes[node]["intensity_list"] = np.stack(nodes[node]["intensity_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        intensity_mean = np.mean(nodes[node]["intensity_list"])
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        
        features = np.concatenate([
            [intensity_mean],
            pos_mean,
        ])
        
        G.add_node(node, features=list(features))

    segments_ids = np.unique(segments)
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])

    # Convert to PyTorch Geometric Data
    x = torch.tensor([G.nodes[node]['features'] for node in G.nodes], dtype=torch.float)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def preprocess_fashion_mnist(root, train=True):
    dataset = datasets.FashionMNIST(root, train=train, download=True)
    
    data_list = []
    edge_slice = [0]
    
    for img, label in tqdm(dataset, desc="Processing images"):
        # Convert PIL Image to numpy array
        img_np = np.array(img).squeeze()
        graph_data = get_graph_from_image(img_np)
        
        data_list.append(graph_data)
        edge_slice.append(edge_slice[-1] + graph_data.edge_index.size(1))
    
    # Concatenate all graphs
    x = torch.cat([data.x for data in data_list], dim=0)
    edge_index = torch.cat([data.edge_index + i * 75 for i, data in enumerate(data_list)], dim=1)
    y = torch.tensor([label for _, label in dataset])
    pos = x[:, 1:]  # Assuming the last two features are position
    
    edge_slice = torch.tensor(edge_slice, dtype=torch.long)
    
    return x, edge_index, edge_slice, pos, y

def save_preprocessed_data(root='data/FashionMNIST'):
    # Ensure the directory exists
    os.makedirs(os.path.join(root, 'raw'), exist_ok=True)

    # Process and save training data
    print("Processing training data...")
    x_train, edge_index_train, edge_slice_train, pos_train, y_train = preprocess_fashion_mnist(root, train=True)
    torch.save((x_train, edge_index_train, edge_slice_train, pos_train, y_train), 
               os.path.join(root, 'raw', 'fashion_training.pt'))
    print("Training data saved.")

    # Process and save test data
    print("Processing test data...")
    x_test, edge_index_test, edge_slice_test, pos_test, y_test = preprocess_fashion_mnist(root, train=False)
    torch.save((x_test, edge_index_test, edge_slice_test, pos_test, y_test), 
               os.path.join(root, 'raw', 'fashion_test.pt'))
    print("Test data saved.")

if __name__ == "__main__":
    save_preprocessed_data()