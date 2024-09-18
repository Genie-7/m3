import argparse
import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torchvision.datasets import FashionMNIST, CIFAR10
import numpy as np
from tqdm import tqdm

from architectures import SGCN

parser = argparse.ArgumentParser()

parser.add_argument('dataset_name', type=str, default='MNISTSuperpixels')
parser.add_argument('--num_epoch', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--train_augmentation', type=bool, default=False)
parser.add_argument('--layers_num', type=int, default=3)
parser.add_argument('--model_dim', type=int, default=16)
parser.add_argument('--out_channels_1', type=int, default=64)
parser.add_argument('--use_cluster_pooling', type=bool, default=True)
parser.add_argument('--dim_coor', type=int, default=2)
parser.add_argument('--label_dim', type=int, default=1)
parser.add_argument('--out_dim', type=int, default=10)

args = parser.parse_args()

print(args.use_cluster_pooling)

class CIFAR10Graph(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super(CIFAR10Graph, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['cifar10_training.pt', 'cifar10_test.pt']

    @property
    def processed_file_names(self):
        return ['processed_cifar10_training.pt', 'processed_cifar10_test.pt']

    def download(self):
        # Download CIFAR10 dataset using torchvision
        dataset = CIFAR10(self.raw_dir, train=True, download=True)
        torch.save(dataset, os.path.join(self.raw_dir, 'cifar10_training.pt'))
        dataset = CIFAR10(self.raw_dir, train=False, download=True)
        torch.save(dataset, os.path.join(self.raw_dir, 'cifar10_test.pt'))

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            dataset = torch.load(raw_path)
            data_list = []
            for img, label in dataset:
                img = np.array(img).astype(np.float32)
                # Normalize the image
                img = img / 255.0
                # Flatten the image (3072, 1) - flattened 32x32x3 image
                x = torch.tensor(img, dtype=torch.float).view(-1, 1)
                y = torch.tensor([label], dtype=torch.long)

                edge_index = self.create_grid_edges(32, 32, 3)  # CIFAR10 images are 32x32x3

                # Create pos attribute (pixel coordinates)
                pos = torch.stack([torch.arange(32).repeat(32*3), 
                                   torch.arange(32).repeat_interleave(32).repeat(3)], dim=1).float()

                data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
                data_list.append(data)
            
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            
            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)

    def create_grid_edges(self, rows, cols, channels):
        edge_index = []
        total_pixels = rows * cols * channels
        for i in range(total_pixels):
            row = (i // channels) // cols
            col = (i // channels) % cols
            channel = i % channels
            
            # Connect to right neighbor
            if col < cols - 1:
                edge_index.append([i, i + channels])
            
            # Connect to bottom neighbor
            if row < rows - 1:
                edge_index.append([i, i + cols * channels])
            
            # Connect to next channel
            if channel < channels - 1:
                edge_index.append([i, i + 1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index


class FashionMNISTGraph(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super(FashionMNISTGraph, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['fashion_mnist_training.pt', 'fashion_mnist_test.pt']

    @property
    def processed_file_names(self):
        return ['processed_fashion_mnist_training.pt', 'processed_fashion_mnist_test.pt']

    def download(self):
        # Download FashionMNIST dataset using torchvision
        dataset = FashionMNIST(self.raw_dir, train=True, download=True)
        torch.save(dataset, os.path.join(self.raw_dir, 'fashion_mnist_training.pt'))
        dataset = FashionMNIST(self.raw_dir, train=False, download=True)
        torch.save(dataset, os.path.join(self.raw_dir, 'fashion_mnist_test.pt'))

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            dataset = torch.load(raw_path)
            data_list = []
            for img, label in dataset:
                img = np.array(img).astype(np.float32).flatten()
                x = torch.tensor(img, dtype=torch.float).view(-1, 1)
                y = torch.tensor([label], dtype=torch.long)

                edge_index = self.create_grid_edges(28, 28)  # FashionMNIST images are 28x28

                # Create pos attribute (pixel coordinates)
                pos = torch.stack([torch.arange(28).repeat(28), 
                                   torch.arange(28).repeat_interleave(28)], dim=1).float()

                data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
                data_list.append(data)
            
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            
            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)

    def create_grid_edges(self, rows, cols):
        edge_index = []
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if j < cols - 1:  # connect right
                    edge_index.append([idx, idx + 1])
                if i < rows - 1:  # connect down
                    edge_index.append([idx, idx + cols])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

class LocalMNISTSuperpixels(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super(LocalMNISTSuperpixels, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return ['processed_training.pt', 'processed_test.pt']

    def download(self):
        # Download is not needed as files are already present
        pass

    def process(self):
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            x, edge_index, edge_slice, pos, y = torch.load(raw_path)
            edge_index, y = edge_index.to(torch.long), y.to(torch.long)
            m, n = y.size(0), 75
            x, pos = x.view(m * n, 1), pos.view(m * n, 2)
            node_slice = torch.arange(0, (m + 1) * n, step=n, dtype=torch.long)
            graph_slice = torch.arange(m + 1, dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
            slices = {
                'x': node_slice,
                'edge_index': edge_slice,
                'y': graph_slice,
                'pos': node_slice
            }
            
            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [d for d in data_list if self.pre_filter(d)]
                data, slices = self.collate(data_list)
            
            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(d) for d in data_list]
                data, slices = self.collate(data_list)
            
            torch.save((data, slices), processed_path)


if args.dataset_name == 'MNISTSuperpixels':
    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'MNIST')
    train_dataset = LocalMNISTSuperpixels(path, True, transform=T.Cartesian())
    test_dataset = LocalMNISTSuperpixels(path, False, transform=T.Cartesian())
elif args.dataset_name == 'FashionMNISTGraph':
    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'FashionMNIST')
    train_dataset = FashionMNISTGraph(path, True, transform=None)
    test_dataset = FashionMNISTGraph(path, False, transform=None)
elif args.dataset_name == 'CIFAR10Graph':
    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'CIFAR10')
    train_dataset = CIFAR10Graph(path, True, transform=None)
    test_dataset = CIFAR10Graph(path, False, transform=None)
else:
    raise ValueError(f"Unknown dataset name {args.dataset_name}")
    

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SGCN(dim_coor=args.dim_coor,
             out_dim=args.out_dim,
             input_features=1,
             layers_num=args.layers_num,
             model_dim=args.model_dim,
             out_channels_1=args.out_channels_1,
             dropout=args.dropout,
             use_cluster_pooling=args.use_cluster_pooling).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# rotation_0 = T.RandomRotate(degrees=180, axis=0)
# rotation_1 = T.RandomRotate(degrees=180, axis=1)
# rotation_2 = T.RandomRotate(degrees=180, axis=2)

def train(epoch):
    model.train()
    loss_all = 0
    
    # Wrap train_loader with tqdm
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=True)
    
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        # if args.train_augmentation:
        #     data = rotation_0(data)
        #     data = rotation_1(data)
        #     data = rotation_2(data)
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        
        # Update progress bar with current loss
    
    return loss_all / len(train_dataset)

# @torch.no_grad()
# def test(loader):
#     model.eval()

#     correct = 0
#     for data in loader:
#         data = data.to(device)
#         pred = model(data).max(dim=1)[1]
#         correct += pred.eq(data.y.squeeze()).sum().item()  # Add .squeeze() here
#     return correct / len(loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()

    correct = 0
    pbar = tqdm(loader, desc="Testing", leave=True)
    for data in pbar:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y.squeeze()).sum().item()  # Add .squeeze() here
    return correct / len(loader.dataset)

train_acc_array = []
test_acc_array = []
best_test_acc = 0

for epoch in range(1, args.num_epoch):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    train_acc_array.append(train_acc)
    test_acc_array.append(test_acc)

    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f} Best Test Acc: {:.5f}'.format(epoch, loss, train_acc, test_acc, best_test_acc))

print('Best test accuracy: {:.5f}'.format(best_test_acc))