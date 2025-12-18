import os
import torch
from torch_geometric.data import Data, Dataset
from PIL import Image
import numpy as np
from skimage import segmentation, color
from skimage.feature import local_binary_pattern
from typing import Optional

class ImageToGraphConverter:
    """Convert images to graphs using superpixels."""
    
    def __init__(self, n_segments: int = 100, compactness: float = 10.0, 
                 n_neighbors: int = 8, use_lbp: bool = True):
        self.n_segments = n_segments
        self.compactness = compactness
        self.n_neighbors = n_neighbors
        self.use_lbp = use_lbp
        
    def _extract_node_features(self, image: np.ndarray, labels: np.ndarray, 
                               region_id: int) -> np.ndarray:
        """Extract features for a single superpixel region."""
        mask = (labels == region_id)
        region = image[mask]
        
        features = []
        
        # Color features (mean and std for each channel)
        if len(region.shape) == 2:  # Grayscale
            features.extend([np.mean(region), np.std(region)])
        else:  # RGB
            for channel in range(region.shape[1]):
                features.extend([np.mean(region[:, channel]), 
                                np.std(region[:, channel])])
        
        # Texture features (Local Binary Pattern)
        if self.use_lbp:
            try:
                lbp = local_binary_pattern(
                    image if len(image.shape) == 2 else color.rgb2gray(image),
                    P=8, R=1, method='uniform'
                )
                lbp_region = lbp[mask]
                features.extend([np.mean(lbp_region), np.std(lbp_region)])
            except:
                features.extend([0.0, 0.0])
        
        # Spatial features (centroid)
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            features.extend([np.mean(y_coords) / image.shape[0], 
                           np.mean(x_coords) / image.shape[1]])
        else:
            features.extend([0.0, 0.0])
        
        # Size feature
        features.append(np.sum(mask) / (image.shape[0] * image.shape[1]))
        
        return np.array(features, dtype=np.float32)
    
    def _create_edges(self, labels: np.ndarray) -> np.ndarray:
        """Create edges between adjacent superpixels."""
        n_regions = len(np.unique(labels))
        
        # Create edges based on spatial adjacency using 4-connectivity
        h, w = labels.shape
        edge_set = set()
        
        # Check vertical neighbors
        for i in range(h - 1):
            for j in range(w):
                r1, r2 = int(labels[i, j]), int(labels[i + 1, j])
                if r1 != r2:
                    edge_set.add((r1, r2))
                    edge_set.add((r2, r1))  # Undirected graph
        
        # Check horizontal neighbors
        for i in range(h):
            for j in range(w - 1):
                r1, r2 = int(labels[i, j]), int(labels[i, j + 1])
                if r1 != r2:
                    edge_set.add((r1, r2))
                    edge_set.add((r2, r1))  # Undirected graph
        
        # If no edges found, create a simple connectivity pattern
        if len(edge_set) == 0:
            # Create a chain or fully connected based on number of regions
            if n_regions == 1:
                edge_set.add((0, 0))  # Self-loop
            else:
                # Create edges between adjacent region indices
                for i in range(n_regions - 1):
                    edge_set.add((i, i + 1))
                    edge_set.add((i + 1, i))
        
        # Convert to numpy array in shape [2, num_edges]
        if len(edge_set) > 0:
            edge_list = list(edge_set)
            edge_index = np.array(edge_list, dtype=np.int64).T
        else:
            # Fallback: self-loops for all nodes
            edge_index = np.array([[i, i] for i in range(n_regions)], dtype=np.int64).T
        
        return edge_index
    
    def convert(self, image: np.ndarray) -> Data:
        """Convert an image to a graph."""
        # Resize image if too large
        max_size = 224
        if image.shape[0] > max_size or image.shape[1] > max_size:
            scale = max_size / max(image.shape[:2])
            new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
            if len(image.shape) == 2:
                image = Image.fromarray(image).resize(new_shape[::-1])
                image = np.array(image)
            else:
                image = Image.fromarray(image).resize(new_shape[::-1])
                image = np.array(image)
        
        # Convert to RGB if needed (keep uint8 to avoid LBP float warnings)
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        elif image.shape[2] == 4:
            image = color.rgba2rgb(image)
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        
        # Create superpixels
        labels = segmentation.slic(
            image, 
            n_segments=self.n_segments, 
            compactness=self.compactness,
            start_label=0
        )
        
        # Extract node features
        n_nodes = len(np.unique(labels))
        node_features = []
        
        for region_id in range(n_nodes):
            features = self._extract_node_features(image, labels, region_id)
            node_features.append(features)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # Normalize features
        node_features = (node_features - node_features.mean(axis=0)) / (node_features.std(axis=0) + 1e-8)
        
        # Create edges
        edge_index = self._create_edges(labels)
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)


class Br35HDataset(Dataset):
    """Dataset loader for br35h dataset."""
    
    def __init__(self, root: str, split: str = 'train', transform=None, 
                 n_segments: int = 100):
        self.root = root
        self.split = split
        self.transform = transform
        self.converter = ImageToGraphConverter(n_segments=n_segments)
        
        self.data_paths = []
        self.labels = []
        
        # Load data based on split
        if split == 'train':
            yes_dir = os.path.join(root, 'yes')
            no_dir = os.path.join(root, 'no')
            
            # Load yes (tumor) images
            if os.path.exists(yes_dir):
                yes_files = sorted([f for f in os.listdir(yes_dir) if f.endswith('.jpg')])
                # Use first 1200 for training
                for f in yes_files[:1200]:
                    self.data_paths.append(os.path.join(yes_dir, f))
                    self.labels.append(1)
            
            # Load no (no tumor) images
            if os.path.exists(no_dir):
                no_files = sorted([f for f in os.listdir(no_dir) if f.endswith('.jpg')])
                # Use first 1200 for training
                for f in no_files[:1200]:
                    self.data_paths.append(os.path.join(no_dir, f))
                    self.labels.append(0)
                    
        elif split == 'test':
            yes_dir = os.path.join(root, 'yes')
            no_dir = os.path.join(root, 'no')
            
            # Load remaining yes (tumor) images for testing
            if os.path.exists(yes_dir):
                yes_files = sorted([f for f in os.listdir(yes_dir) if f.endswith('.jpg')])
                # Use remaining 300 for testing
                for f in yes_files[1200:1500]:
                    self.data_paths.append(os.path.join(yes_dir, f))
                    self.labels.append(1)
            
            # Load remaining no (no tumor) images for testing
            if os.path.exists(no_dir):
                no_files = sorted([f for f in os.listdir(no_dir) if f.endswith('.jpg')])
                # Use remaining 300 for testing
                for f in no_files[1200:1500]:
                    self.data_paths.append(os.path.join(no_dir, f))
                    self.labels.append(0)
        
        super(Br35HDataset, self).__init__(root, transform)
    
    def len(self):
        return len(self.data_paths)
    
    def get(self, idx):
        img_path = self.data_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Convert to graph
        data = self.converter.convert(image)
        data.y = torch.tensor([label], dtype=torch.long)
        data.img_path = img_path
        
        if self.transform:
            data = self.transform(data)
        
        return data


class SartajDataset(Dataset):
    """Dataset loader for sartaj dataset."""
    
    def __init__(self, root: str, split: str = 'train', transform=None,
                 n_segments: int = 100):
        self.root = root
        self.split = split
        self.transform = transform
        self.converter = ImageToGraphConverter(n_segments=n_segments)
        
        self.data_paths = []
        self.labels = []
        
        # Class mapping
        self.class_to_idx = {
            'glioma_tumor': 0,
            'meningioma_tumor': 1,
            'no_tumor': 2,
            'pituitary_tumor': 3
        }
        
        split_dir = 'Training' if split == 'train' else 'Testing'
        base_dir = os.path.join(root, split_dir)
        
        # Load images from each class
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(base_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
                # Use subset for faster training
                max_files = 200 if split == 'train' else 50
                for f in files[:max_files]:
                    self.data_paths.append(os.path.join(class_dir, f))
                    self.labels.append(class_idx)
        
        super(SartajDataset, self).__init__(root, transform)
    
    def len(self):
        return len(self.data_paths)
    
    def get(self, idx):
        img_path = self.data_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Convert to graph
        data = self.converter.convert(image)
        data.y = torch.tensor([label], dtype=torch.long)
        data.img_path = img_path
        
        if self.transform:
            data = self.transform(data)
        
        return data

