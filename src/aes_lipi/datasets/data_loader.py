import argparse
import itertools
import logging
import os
import sys
from typing import List, Optional, Tuple

import requests
import torch
from torch.distributions import binomial, uniform
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np


device = "cpu"
dtype = torch.float32


def create_binary_cluster_data(n_clusters: int=10, n_dim: int=100, noise_probability: float=0.05, p: float=0.5, cluster_centroids: Optional[torch.Tensor]=None, distribution: Optional[binomial.Binomial]=None, n_examples_per_cluster: int=100) -> str:
    logging.info(f"Create binary cluster data with n_dim: {n_dim} and n_clusters: {n_clusters}")
    if distribution is None:
        distribution = binomial.Binomial(2, probs=p)
    if cluster_centroids is None:
        cluster_centroids = distribution.sample((n_clusters, n_dim))
    uniform_d = uniform.Uniform(0, 1)
    mask = (
        uniform_d.sample((n_clusters, n_examples_per_cluster, n_dim))
        < noise_probability
    )
    _n_samples = int(n_examples_per_cluster / n_clusters)
    assert n_examples_per_cluster % n_clusters == 0, f"n_examples_per_cluster must be dividable by n_clusters"
    samples = cluster_centroids.repeat(n_clusters, _n_samples, 1)
    print(f"Cluster centroids: {cluster_centroids.shape}, mask: {mask.shape}, sample: {samples.shape}")
    train_d = torch.logical_xor(samples, mask)
    mask = uniform_d.sample((n_examples_per_cluster, n_dim)) < noise_probability
    test_d = torch.logical_xor(samples, mask)
    mask = uniform_d.sample((n_examples_per_cluster, n_dim)) < noise_probability
    validation_d = torch.logical_xor(samples, mask)

    folder_name = "binary_problem_data"
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"binary_clustering_{n_clusters}_{n_examples_per_cluster}_{n_dim}.pt"
        
    datasets = {"train": train_d, "test": test_d, "validation": validation_d}
    for key, value in datasets.items():
        out_file = os.path.join(folder_name, f"{key}_{file_name}")
        torch.save(value, out_file)
        logging.info(f"Save {out_file}")
        
        out_file = os.path.join(folder_name, f"{key}_T_{file_name}")
        torch.save(value.view(-1, n_dim), out_file)
        logging.info(f"Save {out_file}")
        
    return file_name
    

class BinaryClusterDataset(Dataset):
    def __init__(self, data_file) -> None:
        super(BinaryClusterDataset).__init__()
        self.data = torch.load(data_file).to(torch.float32)
        self.data_file = data_file
        logging.info(f"Loading {self.data.shape} from {data_file}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx, :]
        return sample, idx


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = tensor - min_tensor
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def create_batches(batch_size: int, dataset_name: str, data_path: str="") -> Tuple[DataLoader, DataLoader, int, int]:
    # TODO improve dataset loading
    if dataset_name.startswith("binary_clustering"):
        # Dataset
        n_clusters, n_examples, n_dim = dataset_name.split("_")[-3:]
        train_dataset = BinaryClusterDataset(
            f"{data_path}binary_problem_data/train_{dataset_name}.pt"
        )
        test_dataset = BinaryClusterDataset(
            f"{data_path}binary_problem_data/test_{dataset_name}.pt"
        )
        width = int(n_dim)
        height = 1
    else:
        raise Exception(f"Unknown data set: {dataset_name}")        

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    logging.info(f"Create datasets with batch size {batch_size}")
    return train_loader, test_loader, width, height


def parse_arguments(param: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments (`sys.argv`).
    """
    parser = argparse.ArgumentParser(description="Data creation for Lipi-AE")
    parser.add_argument(
        "--n_dim",
        type=int,
        default=100,
        help="Binary cluster data number of dimensions. E.g. 100",
    ),
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Binary cluster data number of dimensions. E.g. 10",
    )
    parser.add_argument(
        "--noise_probability",
        type=float,
        default=0.05,
        help="Noise probability. E.g. 0.05",
    )
    parser.add_argument(
        "--n_examples_per_cluster",
        type=int,
        default=100,
        help="Number of examples per cluster. E.g. 100",
    )
    args = parser.parse_args(param)
    return args

if __name__ == "__main__":
    log_file = os.path.basename(__file__).replace(".py", ".log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    args = parse_arguments(sys.argv[1:])
    create_binary_cluster_data(n_dim=args.n_dim, n_clusters=args.n_clusters, noise_probability=args.noise_probability, n_examples_per_cluster=args.n_examples_per_cluster)
