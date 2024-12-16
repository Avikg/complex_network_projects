import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support as prfs_support
import time
import argparse
from math import sqrt
import os

class SparseToTensor:
    """Utility class for converting sparse matrices to PyTorch tensors."""
    @staticmethod
    def convert(sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

class NormalizeFeatures:
    """Method for normalizing feature matrices."""
    @staticmethod
    def normalize(mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_diag = sp.diags(r_inv)
        mx_normalized = r_mat_diag.dot(mx)
        return mx_normalized

class DataPreparation:
    """Class to handle loading and preparing data."""
    @staticmethod
    def load_data(path="cora"):
        # Get the absolute path of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up one directory from the current script location
        parent_dir = os.path.dirname(current_dir)
        # Construct the dataset paths
        content_path = os.path.join(parent_dir, f"{path}.content")
        cites_path = os.path.join(parent_dir, f"{path}.cites")

        print(f'Loading {path} dataset from: {content_path}')

        content = np.genfromtxt(content_path, dtype=np.dtype(str))
        features = sp.csr_matrix(content[:, 1:-1], dtype=np.float32)
        labels = DataPreparation.encode_labels(content[:, -1])

        idx = np.array(content[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        
        edges_unordered = np.genfromtxt(cites_path, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = NormalizeFeatures.normalize(features)
        adj = NormalizeFeatures.normalize(adj + sp.eye(adj.shape[0]))

        return SparseToTensor.convert(adj), torch.FloatTensor(np.array(features.todense())), torch.LongTensor(np.where(labels)[1])

    @staticmethod
    def encode_labels(labels):
        classes = set(labels)
        class_dict = {c: np.eye(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_encoded = np.array(list(map(class_dict.get, labels)), dtype=np.int32)
        return labels_encoded

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias if self.bias is not None else output

class GCNModel(nn.Module):
    def __init__(self, nfeat, nclass, nhid=16, dropout=0.5):
        super(GCNModel, self).__init__()
        self.layer1 = GCNLayer(nfeat, nhid)
        self.layer2 = GCNLayer(nhid, nhid)
        self.layer3 = GCNLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.layer1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer3(x, adj)
        return F.log_softmax(x, dim=1)

def evaluate_model(model, features, adj, labels, idx):
    model.eval()
    output = model(features, adj)
    loss = F.nll_loss(output[idx], labels[idx])
    preds = output.max(1)[1].type_as(labels)
    accuracy = preds.eq(labels).double().sum() / len(labels)
    labels_actual = labels.cpu().numpy()
    preds_actual = preds.cpu().numpy()
    precision, recall, f1, _ = prfs_support(labels_actual, preds_actual, average='weighted')
    
    metrics = f"Test Loss: {loss.item():.4f}, Test Acc: {accuracy.item():.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    
    # Ensure the GCN directory exists
    # metrics_dir = 'GCN'
    # if not os.path.exists(metrics_dir):
    #     os.makedirs(metrics_dir)

    # Define the path for the metrics file within the GCN directory
    metrics_file_path = "gcn_metrics.txt"

    with open(metrics_file_path, 'w+') as file:
        file.write(metrics)
    
    print(f"Test results written to {metrics_file_path}")

def main():
    parser = argparse.ArgumentParser(description='GCN Implementation')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')

    args = parser.parse_args(args=[])
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    adj, features, labels = DataPreparation.load_data("cora")
    model = GCNModel(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.nll_loss(output, labels)
        acc = (output.max(1)[1] == labels).float().mean()
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}')

    evaluate_model(model, features, adj, labels, np.arange(len(labels)))

if __name__ == "__main__":
    main()
