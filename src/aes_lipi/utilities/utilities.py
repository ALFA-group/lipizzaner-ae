import argparse
import importlib
import json
import logging
from dataclasses import dataclass, field

import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import seaborn as sns

from aes_lipi.anns.ann import Autoencoder, Decoder, Encoder, VariationalAutoencoder
from aes_lipi.datasets.data_loader import create_batches
from aes_lipi.environments.binary_clustering import (
    AutoencoderBinaryClustering,
    DecoderBinaryClustering,
    DenoisingAutoencoderBinaryClustering,
    EncoderBinaryClustering,
    VariationalAutoencoderBinaryClustering,
    VariationalDecoderBinaryClustering,
    VariationalEncoderBinaryClustering,
)


dtype = torch.float32
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


@dataclass
class Node:
    """Lipizzaner node autoencoder"""

    # Classes
    Encoder: torch.nn.Module
    Decoder: torch.nn.Module
    Autoencoder: torch.nn.Module
    # Instances
    index: int
    learning_rate: float
    kwargs: Dict[str, Any]
    training_data: Dataset
    optimizer: Dict[str, Any]
    encoders: List[torch.nn.Module] = field(default_factory=list)
    decoders: List[torch.nn.Module] = field(default_factory=list)


def load_data_and_autoencoder(batch_size: int, ann_path: str, autoencoder: str, dataset_name: str, data_loader_str: str="test", data_path: str="") -> Tuple[DataLoader, DataLoader, int, int, Any]:
    train_loader, test_loader, width, height = create_batches(batch_size, dataset_name, data_path)
    if data_loader_str == "test":
        data_loader = test_loader
    else:
        data_loader = train_loader
        
    data = torch.load(ann_path)
    Autoencoder, Encoder, Decoder, kwargs = get_autoencoder(autoencoder, data_loader, width, height)
    encoder = Encoder(**kwargs)
    encoder = encoder.to(device)
    encoder.load_state_dict(data["encoder"])
    decoder = Decoder(**kwargs)
    decoder = decoder.to(device)
    decoder.load_state_dict(data["decoder"])
    ae = Autoencoder(encoder, decoder)
    ae = ae.to(device)
    return train_loader, test_loader, width, height, ae
    
def measure_ae_quality(
    ann_path: str, autoencoder: str, batch_size: int, dataset_name: str
) -> Dict[str, Any]:
    logging.info(f"Measure AE quality for {ann_path} {autoencoder} {batch_size}")
    # TODO hardcoded...
    train_loader, test_loader, width, height, ae = load_data_and_autoencoder(batch_size, ann_path, autoencoder, dataset_name, data_loader_str="test")
    errors = measure_reconstruction_error(ae, test_loader)
    return {"L1": errors}


def measure_reconstruction_error(ae: torch.nn.Module, data_loader: torch.Tensor) -> float:
    errors = []
    l1_loss = torch.nn.L1Loss()
    with torch.no_grad():
        for _, (data, _) in enumerate(data_loader):
            if torch.cuda.is_available():
                data = data.to("cuda")
            if isinstance(data, list):
                # TODO ugly hack
                data = data[0]
            data = data.view(-1, ae.x_dim)
            if isinstance(ae, VariationalAutoencoder):
                recon_batch, mu, log_var = ae(data)
            else:
                recon_batch = ae(data)

            errors.append(l1_loss(recon_batch, data).data.item())
    
    errors = np.asarray(errors).reshape((len(errors), 1))
    return errors
        

def measure_quality_scores(scores: List[str], ae: torch.nn.Module, data_loader: torch.Tensor) -> Dict[str, float]:
    #TODO ugly
    score_fcn_map = {"L1": measure_reconstruction_error}
    if 'all' in scores:
        scores = list(score_fcn_map.keys())
        logging.debug(f"Using all scores from {scores}")
        
    quality_scores = {}
    for score in scores:
        score_fcn = score_fcn_map.get(score)
        value = score_fcn(ae, data_loader)
        key = f"mean_{score}"
        quality_scores[key] = np.mean(value)
        key = f"std_{score}"
        quality_scores[key] = np.std(value)
        
    return quality_scores
    
def plot_stats(stats: List[Dict[str, Any]], output_dir: str, id: str="NA"):
    df = pd.DataFrame(stats)
    print(df.head())
    plt.clf()
    # Learning rates
    sns.lineplot(data=df, x="iteration", y="learning_rate", hue="node_idx")
    out_path = os.path.join(output_dir, "learning_rates.png")
    plt.savefig(out_path)
    plt.clf()
    # Losses
    sns.lineplot(data=df, x="iteration", y="min_replacement_loss", hue="node_idx")
    out_path = os.path.join(output_dir, "replacement_loss.png")
    plt.savefig(out_path)
    plt.clf()
    sns.lineplot(
        data=df, x="iteration", y="min_selection_loss", style="node_idx", markers=True
    )
    out_path = os.path.join(output_dir, "selection_loss.png")
    plt.savefig(out_path)
    plt.close()
    plt.clf()
    # Losses
    sns.lineplot(data=df, x="iteration", y="min_replacement_loss")
    out_path = os.path.join(output_dir, "pop_replacement_loss.png")
    plt.savefig(out_path)
    plt.clf()
    sns.lineplot(
        data=df, x="iteration", y="min_selection_loss", markers=True
    )
    out_path = os.path.join(output_dir, "pop_selection_loss.png")
    plt.savefig(out_path)
    plt.close()
    out_file = os.path.join(output_dir, "stats.jsonl")
    with open(out_file, "w") as fd:
        for stat in stats:
            # TODO more meaningful ID
            stat["id"] = str(id)
            json.dump(stat, fd)
            fd.write("\n")

    logging.info(f"Wrote stats to {out_file}")


def plot_node_losses(
    losses: np.ndarray, node_idx: int, iteration: int, note: str, output_dir: str
):
    # Use the ae loss
    losses = losses[:, :, 0]
    fig, (ax_0, ax_1, ax_2) = plt.subplots(nrows=1, ncols=3)
    fig.suptitle(f"Losses N {node_idx} t {iteration} {note}")
    ax_0.set_title("All losses")
    ax_0.matshow(losses)
    for (i, j), z in np.ndenumerate(losses):
        ax_0.text(j, i, "{:0.1f}".format(z), ha="center", va="center")

    ax_1.set_title("Max losses")
    max_losses = np.reshape(np.max(losses, axis=1), (-1, 1))
    ax_1.matshow(max_losses)
    for (i, j), z in np.ndenumerate(max_losses):
        ax_1.text(j, i, "{:0.1f}".format(z), ha="center", va="center")

    ax_2.set_title("Mean losses")
    mean_losses = np.reshape(np.mean(losses, axis=1), (-1, 1))
    ax_2.matshow(mean_losses)
    for (i, j), z in np.ndenumerate(mean_losses):
        ax_2.text(j, i, "{:0.1f}".format(z), ha="center", va="center")

    out_file = os.path.join(
        output_dir, f"loss_t{iteration}_n{node_idx}_{note}_lipi_ae.png"
    )
    plt.savefig(out_file)
    plt.close()


def plot_train_data(
    node: Node,
    batch: torch.Tensor,
    iteration: int,
    Autoencoder: torch.nn.Module,
    output_dir: str,
):
    ae = Autoencoder(node.encoders[0], node.decoders[0])
    note = f"train_t{iteration}_n{node.index}"
    plot_ae_data(ae, batch, note, output_dir)


def plot_ae_data(ae: torch.nn.Module, data: torch.Tensor, note: str, output_dir: str):
    # TODO improve fige figure sizes...
    plt.figure(figsize=(1, 1), dpi=300)
    if isinstance(data, list):
        data = data[0]
    x = data.view(-1, ae.x_dim)
    if isinstance(ae, VariationalAutoencoder):
        recon, mu, log_var = ae(x)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(log_var))
        z = p.rsample((1,))
        im = ae.decoder.decode(z)
    else:
        recon = ae(x)
        im = recon.view(recon.size(0), 1, ae.height, ae.width)
        
    im_path = os.path.join(output_dir, f"sample_{note}_lipi_ae.png")
    save_image(im.view(recon.size(0), 1, ae.height, ae.width), im_path)
    plt.close()


def show_images(decoder: torch.nn.Module, output_dir: str) -> None:
    plt.figure(figsize=(1, 1), dpi=300)
    z = torch.randn((256, decoder.z_dim)).to(device=device)
    im_path = os.path.join(output_dir, "sample_rnd_lipi_ae.png")
    with torch.no_grad():
        im = decoder.decode(z.to(device)).cpu()
        im_shape = (z.size(0), 1, decoder.height, decoder.width)            
            
    im = im.view(*im_shape)
    save_image(im.view(*im_shape), im_path)    
    logging.info(f"Saved decoded noise to {im_path}")
    plt.close()


def analyze_data(file_path: str):
    data = []
    with open(file_path, "r") as fd:
        for line in fd:
            data.append(json.loads(line))
    output_dir = os.path.dirname(file_path)
    plot_stats(data, output_dir)


def load_autoencoder(file_path: str, environment: str):
    # TODO test
    Autoencoder, Encoder, Decoder, kwargs = get_autoencoder(environment, training_data=None, width=-1, height=-1)
    encoder = Encoder(**kwargs).to(device)
    decoder = Decoder(**kwargs).to(device)
    data = torch.load(file_path)
    encoder.load_state_dict(data["encoder"])
    decoder.load_state_dict(data["decoder"])
    autoencoder = Autoencoder(encoder, decoder).to(device)
    
    return autoencoder
                
def get_module(name):
    # TODO do I need this?
    module_str, _ = name.rsplit(".", 1)
    module = importlib.import_module(module_str)
    return module


def get_autoencoder(environment: str, training_data: DataLoader, width: int, height: int) -> Tuple[Autoencoder, Encoder, Decoder, Dict[str, Any]]:
    # TODO messy still
    logging.info(f"Get environment for {environment}")
    x_dim = width * height
    kwargs = {"width": width, "height": height, "x_dim": x_dim}
    if environment == "VariationalAutoencoderBinaryClustering":
        Autoencoder = VariationalAutoencoderBinaryClustering
        Decoder = VariationalDecoderBinaryClustering
        Encoder = VariationalEncoderBinaryClustering
    elif environment == "AutoencoderBinaryClustering":
        Autoencoder = AutoencoderBinaryClustering
        Decoder = DecoderBinaryClustering
        Encoder = EncoderBinaryClustering
    elif environment == "DenoisingAutoencoderBinaryClustering":
        Autoencoder = DenoisingAutoencoderBinaryClustering
        Decoder = DecoderBinaryClustering
        Encoder = EncoderBinaryClustering
    else:
        raise Exception(f"Undefined environment {environment}")

    logging.info(f"Loading {Autoencoder}, {Encoder}, {Decoder} from {environment}")
    return Autoencoder, Encoder, Decoder, kwargs


def get_dataset_sample(dataloader: DataLoader, fraction: Optional[float]) -> DataLoader:
    if fraction == 0.0:
        logging.info("Not subsampling data")
        return dataloader
    
    n_samples = int(len(dataloader.dataset) * fraction)
    samples = random.choices(range(len(dataloader.dataset)), k=n_samples)
    dataset_sample = torch.utils.data.Subset(dataloader.dataset, samples)
    node_data_loader = torch.utils.data.DataLoader(dataset=dataset_sample, batch_size=dataloader.batch_size, shuffle=True)
    assert len(dataloader) >= len(node_data_loader)
    logging.info(f"Sampled {n_samples}/{len(dataloader.dataset)} from {dataloader.__class__.__name__}")
    return node_data_loader
        

def set_rng_seed(rng_seed: Optional[int]) -> int:
    if rng_seed is None:
        rng_seed = random.randint(0, 2**32 - 1)

    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    logging.info(f"RNG_SEED {rng_seed}")
    return rng_seed


def parse_arguments(param: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AE ANN utils")
    parser.add_argument(
        "--fid",
        type=str,
        default="",
        help="TODO HACK. AE path,Environment. E.g. out/mlp_img/at.pt,AutoencoderMNIST,mnist",
    )
    args = parser.parse_args(param)
    return args


def l2_norm_diff(e0, e1) -> float:
    diffs = {}
    for name, _ in e0.named_parameters():
        param0 = e0.get_parameter(name)
        param1 = e1.get_parameter(name)
        if param0.requires_grad:
            l2_norm_diff = torch.linalg.vector_norm(param0 - param1, ord=2)
            diffs[name] = l2_norm_diff.data.item()
            
    mean_l2_norm_diff = np.mean(list(diffs.values()))
    return mean_l2_norm_diff


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    elif v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise ValueError(v)
    
    
if __name__ == "__main__":
    log_file = os.path.basename(__file__).replace(".py", ".log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    args = parse_arguments(sys.argv[1:])
    analyze_data(os.path.join("", "stats.jsonl"))
