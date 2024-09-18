import argparse
import os
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from architectures import SGCN
import itertools
import json
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T

rotation_0 = T.RandomRotate(degrees=180, axis=0)
rotation_1 = T.RandomRotate(degrees=180, axis=1)
rotation_2 = T.RandomRotate(degrees=180, axis=2)

# Copy LocalMNISTSuperpixels class here
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
# Copy train and test functions here
def train(epoch, model, device, train_loader, optimizer, train_augmentation):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if train_augmentation:
            data = rotation_0(data)
            data = rotation_1(data)
            data = rotation_2(data)
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset)

# Modify test function
def test(loader, model, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def train_and_evaluate(params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SGCN(dim_coor=params['dim_coor'],
                 out_dim=params['out_dim'],
                 input_features=params['label_dim'],
                 layers_num=params['layers_num'],
                 model_dim=params['model_dim'],
                 out_channels_1=params['out_channels_1'],
                 dropout=params['dropout'],
                 use_cluster_pooling=params['use_cluster_pooling']).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'MNIST')
    train_dataset = LocalMNISTSuperpixels(path, True, transform=T.Cartesian())
    test_dataset = LocalMNISTSuperpixels(path, False, transform=T.Cartesian())
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    best_test_acc = 0
    for epoch in range(1, params['num_epoch'] + 1):
        train(epoch, model, device, train_loader, optimizer, params['train_augmentation'])
        train_acc = test(train_loader, model, device)
        test_acc = test(test_loader, model, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    
    return best_test_acc

# Define parameter ranges to search
param_grid = {
    'num_epoch': [100, 200, 300],
    'lr': [0.0001, 0.0005, 0.001],
    'batch_size': [64, 128, 256],
    'dropout': [0.2, 0.3, 0.4],
    'train_augmentation': [False, True],
    'layers_num': [2, 3, 4],
    'model_dim': [16, 32, 64],
    'out_channels_1': [32, 64, 128],
    'dim_coor': [2],  # This is fixed for MNIST
    'label_dim': [1],  # This is fixed for MNIST
    'out_dim': [10],  # This is fixed for MNIST
    'use_cluster_pooling': [True, False]
}

# Generate all combinations of parameters
param_combinations = list(itertools.product(*param_grid.values()))

results = []

for params in tqdm(param_combinations):
    current_params = dict(zip(param_grid.keys(), params))
    accuracy = train_and_evaluate(current_params)
    results.append({'params': current_params, 'accuracy': accuracy})
    
    # Save results after each iteration
    with open('hyperparameter_search_results.json', 'w') as f:
        json.dump(results, f)

# Find best parameters
best_result = max(results, key=lambda x: x['accuracy'])
print(f"Best parameters: {best_result['params']}")
print(f"Best accuracy: {best_result['accuracy']}")