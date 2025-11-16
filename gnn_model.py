import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Batch


class BrainTumorGNN(nn.Module):
    """Graph Neural Network for brain tumor classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2,
                 num_layers: int = 3, dropout: float = 0.5, use_gat: bool = False):
        super(BrainTumorGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat = use_gat
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if use_gat:
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout, concat=False))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if use_gat:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=False))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if use_gat:
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False))
        else:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # *3 for mean, max, sum pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes]
        """
        # Graph convolutional layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling (combine multiple pooling strategies)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        # Concatenate pooling results
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Classification
        out = self.classifier(x)
        
        return out


class GraphClassifier:
    """Wrapper class for training and evaluating the GNN model."""
    
    def __init__(self, model: nn.Module, device: torch.device, scaler=None, grad_clip=None):
        self.model = model.to(device)
        self.device = device
        self.scaler = scaler  # For mixed precision training
        self.grad_clip = grad_clip  # Gradient clipping value
    
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y.squeeze())
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip is not None and self.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y.squeeze())
                
                loss.backward()
                
                # Gradient clipping
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += pred.eq(batch.y.squeeze()).sum().item()
            total += batch.y.size(0)
        
        return total_loss / len(train_loader), correct / total
    
    def evaluate(self, test_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y.squeeze())
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += pred.eq(batch.y.squeeze()).sum().item()
                total += batch.y.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.squeeze().cpu().numpy())
        
        return total_loss / len(test_loader), correct / total, all_preds, all_labels

