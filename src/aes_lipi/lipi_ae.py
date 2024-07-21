import datetime
import itertools
import json
import os
import sys
import argparse
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from aes_lipi.datasets.data_loader import create_batches

from aes_lipi.utilities.utilities import (
    get_autoencoder,
    get_dataset_sample,
    measure_quality_scores,
    plot_ae_data,
    plot_node_losses,
    plot_stats,
    Node,
    plot_train_data,
    set_rng_seed,
    show_images,
    str2bool,
)


dtype = torch.float32
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

loss_report_interval = 100

rng = np.random.default_rng()


def parse_arguments(param: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments (`sys.argv`).
    """
    parser = argparse.ArgumentParser(description="Run AE ANN")
    parser.add_argument(
        "--configuration_file",
        type=str,
        help="JSON configuration file. E.g. " "configurations/demo_lipi_ae.json",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default="none",
        choices=("all", "final", "none"),
        help="Visualize ANNs. E.g. all",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset that will be used. E.g. mnist",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10000,
        help="Checkpoint ANN interval. E.g. 10",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        help="Rng seed. E.g. 10",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=1,
        help="Neighborhood radius. E.g. 1",
    )
    parser.add_argument(
        "--ann_path",
        type=str,
        default="",
        help="Path to load ANNs. IDX in path will be replaced by node index. E.g. checkpoints/nIDX_ae.pt",
    )
    parser.add_argument(
        "--cell_evaluation",
        type=str,
        default="epoch-node",
        choices=("epoch-node", "all_vs_all", "ann_canonical"),
        help="Evaluation of cells. E.g. epoch-node",
    )
    parser.add_argument(
        "--ae_quality_measures",
        type=str,
        default="all",
        help="Comma separated list of ae quality measures. 'all' means all measures. E.g. FID,SSIM",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out_data",
        help="Output directory. Timestamp is appended. E.g. mlp_img",
    )
    parser.add_argument(
        "--environment",
        type=str,
        help="Environment as instanciated from utilities. E.g. AutoencoderBinaryClustering",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["info", "debug", "warning"],
        default="info",
        help="Loglevel, e.g. --log_level info",
    )
    parser.add_argument(
        "--solution_concept",
        type=str,
        choices=["worst_case", "best_case", "mean_expected_utility"],
        default="worst_case",
        help="Solution concept used to rate AEs, e.g. --solution_concept worst_case",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=2,
        help="Population size. E.g. 2",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs. E.g. 2",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4000,
        help="Batch size. E.g. 2",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate. E.g. 0.0005",
    )
    parser.add_argument(
        "--no_execution",
        action="store_true",
        help="Do not run. Used for testing",
    )
    parser.add_argument(
        "--do_not_overwrite_checkpoint",
        action="store_false",
        default=True,
        help="Do not overwrite checkpoint",
        dest="overwrite_checkpoint",
    )
    parser.add_argument(
        "--store_all_nodes",
        action="store_true",
        default=False,
        help="Store all nodes. Useful for creating ensembles"
    )
    parser.add_argument(
        "--calculate_test_loss",
        type=str2bool,
        default=False,
        help="Calculate test set loss each epoch"
    )    

    args = parser.parse_args(param)
    return args


def get_best_nodes(nodes: List[Node]) -> List[Node]:
    logging.warning("IMPLEMENT GET BEST NODES")
    return nodes[0], nodes[0]


def get_opponent(anns: List[torch.nn.Module]) -> torch.nn.Module:
    opponent = rng.choice(anns)
    logging.debug(f"Opponent {opponent.__class__.__name__}")
    return opponent


def update_learning_rate(learning_rate: float, step_std=0.00001, **kwargs) -> float:
    # TODO todo use pytorch scheduler. Seems like could be more SE to get it to work with optimizers
    # TODO better setting of step size
    MIN_LR = 0.00001 if kwargs.get("sgd_optimizer", False) else 0.000000001
    MAX_LR = 10
    delta = rng.normal(0, step_std)
    learning_rate = max(MIN_LR, learning_rate + delta)
    logging.debug(f"lr: {learning_rate}, delta: {delta}")
    # TODO lower learning rate to avoid errors
    learning_rate = min(MAX_LR, learning_rate)
    
    logging.debug(f"Update learning rate to {learning_rate} original {kwargs.get('original_learning_rate')}")
    return learning_rate


def get_neighbors(nodes: Dict[int, Node], node: Node, radius: int) -> Node:
    encoders = [node.encoders[0]]
    decoders = [node.decoders[0]]
    idx = [node.index]
    for i in range(1, radius + 1):
        # Right
        r_idx = (node.index + i) % len(nodes)
        neighbor = nodes[r_idx].encoders[0]
        encoders.append(neighbor)
        neighbor = nodes[r_idx].decoders[0]
        decoders.append(neighbor)
        l_idx = (node.index - i) % len(nodes)
        # TODO reduce copying? (Only selected networks are modified, and they are copied)
        # Left
        neighbor = nodes[l_idx].encoders[0]
        encoders.append(neighbor)
        neighbor = nodes[l_idx].decoders[0]
        decoders.append(neighbor)
        idx.extend((r_idx, l_idx))

    node.encoders = encoders
    node.decoders = decoders
    assert len(node.decoders) == len(node.encoders) == ((2 * radius) + 1)
    logging.debug(
        f"Get {radius * 2} neighbors ({idx}) from {len(nodes)} nodes for node {node.index}"
    )
    
    return node


def initialize_nodes(
    learning_rate: float, population_size: int, environment: str, ann_path: str, training_data: DataLoader, width: int, height: int, dataset_fraction: float=0.0, identical_initial_ann: bool=False
) -> Dict[int, Node]:
    nodes = {}
    # TODO ugly. Maybe make node a normal class instead of a data class. Also look at Lipi SE
    Autoencoder, Encoder, Decoder, kwargs = get_autoencoder(environment, training_data, width, height)
    dataset = get_dataset_sample(training_data, dataset_fraction)
    logging.info(f"Node kwargs: {kwargs}")
    for i in range(population_size):
        encoder = Encoder(**kwargs)
        decoder = Decoder(**kwargs)        
        if identical_initial_ann and i > 0:
            encoder = nodes[0].encoders[0].clone()
            decoder = nodes[0].decoders[0].clone()
            logging.info(f"{identical_initial_ann} cloning ANNs from 0 to {i}")
            
        node = Node(
            learning_rate=learning_rate,
            index=i,
            encoders=[encoder],
            decoders=[decoder],
            # TODO remove Encoder, Decoder fields and take them from Autoencoder instead?
            Encoder=Encoder,
            Decoder=Decoder,
            Autoencoder=Autoencoder,
            kwargs = kwargs,
            training_data=dataset,
            optimizer=None
        )
        node.encoders[0] = node.encoders[0].to(device)
        node.decoders[0] = node.decoders[0].to(device)
        nodes[i] = node
        if ann_path != "":
            file_path = ann_path.replace("IDX", str(i))
            data = torch.load(file_path)
            nodes[i].encoders[0].load_state_dict(data["encoder"])
            nodes[i].decoders[0].load_state_dict(data["decoder"])
            logging.debug(f"Loading ANNs from {file_path}")
            
    logging.debug(
        f"Initialized {population_size} nodes with learning rate {learning_rate} from {environment}"
    )
    return nodes


def evaluate_cells_epochs_nodes(
    nodes: Dict[int, Node],
    epochs: int,
    visualize: str,
    stats: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_interval: int,
    radius: int,
    score_keys: List[str],
    solution_concept: callable,
    **kwargs: Dict[str, Any],
):
    logging.info("Evaluate cells epoch outer-loop and node inner-loop")
    order = list(range(0, len(nodes)))
    fe_cnt = 0
    for t in range(epochs):
        assert not set(order).symmetric_difference(set(range(0, len(nodes)))), f"{order}"
        for idx in order:
            # Get neighborhood
            node = get_neighbors(nodes, nodes[idx], radius)
                
            # Cell evaluation
            node = evaluate_cell(
                node,
                node.learning_rate,
                iteration=t,
                visualize=visualize,
                stats=stats,
                output_dir=output_dir,
                checkpoint_interval=checkpoint_interval,
                score_keys=score_keys,
                last_iteration=(epochs - 1),
                solution_concept=solution_concept,
                **kwargs
            )
            stats[-1]["fe_cnt"] = fe_cnt
            fe_cnt += 1
            # Update learning rate
            node.learning_rate = update_learning_rate(node.learning_rate, **kwargs)     

        np.random.shuffle(order)
        logging.debug(f"Shuffle cell evaluation order for epoch {t} to {order}")


def evaluate_cells_all_vs_all(
    nodes: Dict[int, Node],
    epochs: int,
    visualize: str,
    stats: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_interval: int,
        score_keys: List[str],
        solution_concept: callable,
            **kwargs: Dict[str, Any],
) -> Dict[str, Node]:
    logging.info(f"Evaluate cell epoch outer-loop and node inner-loop all vs all")
    order = list(range(0, len(nodes)))
    # TODO clunky creating only 1 node
    node_0 = Node(
        learning_rate=nodes[0].learning_rate,
        index=0,
        encoders=[None],
        decoders=[None],
        Encoder=nodes[0].Encoder,
        Decoder=nodes[0].Decoder,
        Autoencoder=nodes[0].Autoencoder
    )
    encoders = [_.encoders[0] for _ in nodes.values()]
    decoders = [_.decoders[0] for _ in nodes.values()]
    fe_cnt = 0
    for t in range(epochs):
        pairs = itertools.product(order, order)
            
        for pair in pairs:
            idx_i, idx_j = pair
            encoder = encoders[idx_i]
            node_0.encoders[0] = encoder
            decoder = decoders[idx_j]
            node_0.decoders[0] = decoder                
            assert len(node_0.encoders) == 1 and len(node_0.decoders) == 1
            # Cell evaluation
            node_0 = evaluate_cell(
                node_0,
                node_0.learning_rate,
                iteration=t,
                visualize=visualize,
                stats=stats,
                output_dir=output_dir,
                checkpoint_interval=checkpoint_interval,
                score_keys=score_keys,
                last_iteration=(epochs - 1),
                solution_concept=solution_concept,
                **kwargs
            )
            stats[-1]["fe_cnt"] = fe_cnt
            fe_cnt += 1
        # Update learning rate
        
        node_0.learning_rate = update_learning_rate(node_0.learning_rate, **kwargs)
        np.random.shuffle(order)
        logging.debug(f"Shuffle cell evaluation order for epoch {t} to {order}")
        logging.info(f"Epoch {t} Min selection loss: {stats[-1]['min_selection_loss']}")

    return {0: node_0}
    

def evaluate_ann_canonical(
    nodes: Dict[int, Node],
    epochs: int,
    visualize: str,
    stats: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_interval: int,
    score_keys: List[str],
    **kwargs: Dict[str, Any]
) -> Dict[str, Node]:
    logging.info(f"Evaluate ann canonical")
    assert len(nodes) == 1
    training_data = nodes[0].training_data
    # TODO is this subpoptimal use of the optimizer?
    ae = nodes[0].Autoencoder(nodes[0].encoders[0], nodes[0].decoders[0])
    ae = ae.to(device)
    learning_rate = nodes[0].learning_rate
    optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs / 4), gamma=0.1)

    fe_cnt = 0
    ae.train()        
    for t in range(epochs):
        stat = {
            "iteration": t,
            "learning_rate": learning_rate,
            "node_idx": nodes[0].index,
            "fe_cnt": fe_cnt,
        }
        losses = []
        for i, (batch, _) in enumerate(training_data):
            batch = batch.to(device)        
            optimizer.zero_grad()        
            loss = ae.compute_loss_together(batch.view(-1, ae.x_dim))
            loss.backward()            
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())

        fe_cnt += 1
        stat["min_selection_loss"] = float(np.min(losses))
        stat["min_replacement_loss"] = float(np.min(losses))
        stat["timestamp"] = float(datetime.datetime.timestamp(datetime.datetime.now()))
        if t % loss_report_interval == 0 or\
            t == (epochs - 1):
                logging.info(f"{t} at loss report interval. {nodes[0].Encoder}")                    
                logging.info(f"{t} calculating quality scores on training_data")                    
                score_values = measure_quality_scores(score_keys, ae, training_data) 
                stat.update(score_values)
                
        if t % checkpoint_interval == 0:
            # NOTE overwrites checkpoints
            out_path = os.path.join(output_dir, "checkpoints", f"n{nodes[0].index}_ae.pt")
            if not kwargs.get("overwrite_checkpoint", True):
                out_path = os.path.join(output_dir, "checkpoints", f"t_{t}_n{nodes[0].index}_ae.pt")
        
            data = {
                "encoder": nodes[0].encoders[0].state_dict(),
                "decoder": nodes[0].decoders[0].state_dict(),
            }
            torch.save(data, out_path)
            logging.debug(f"Checkpoint ANNs for {nodes[0].index} in {out_path}")

        # Update learning rate
        if not kwargs.get("constant_learning_rate", False):
            scheduler.step()
        stats.append(stat)
        learning_rate = scheduler.get_lr()[0]
        logging.info(f"Epoch {t} Min selection loss: {stats[-1]['min_selection_loss']}")
    
    return nodes
    

def main(
    population_size: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    visualize: str,
    cell_evaluation: str,
    environment: str,
    dataset_name: str,
    output_dir: str,
    checkpoint_interval: int,
    radius: int,
    **kwargs,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    kwargs["original_learning_rate"] = learning_rate
    rng_seed = kwargs.get("rng_seed", None)
    solution_concept_str = kwargs.get("solution_concept", None)
    assert solution_concept_str is not None, f"{kwargs}"
    del kwargs["solution_concept"]
    this_mod = sys.modules[__name__]
    solution_concept = getattr(this_mod, solution_concept_str)
    
    score_keys = [_.strip() for _ in kwargs.get("ae_quality_measures", "all").split(",")]
    logging.info(
        f"Run with pop size:{population_size} epochs:{epochs} lr:{learning_rate} bs:{batch_size} sc:{solution_concept_str} {rng_seed} using {device}"
    )
    stats = []
    # TODO store the used seed in the params
    set_rng_seed(rng_seed)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    # Create Batches
    training_data, test_data, width, height = create_batches(batch_size, dataset_name)
    if kwargs.get("calculate_test_loss", False):
        kwargs["test_data"] = test_data
    # Create graph
    nodes = initialize_nodes(
        learning_rate, population_size, environment, kwargs.get("ann_path", ""), training_data, width, height, dataset_fraction=kwargs.get("data_subsample", 0.0), identical_initial_ann=kwargs.get("identical_initial_ann", False)
    )
    if cell_evaluation == "epoch-node":
        evaluate_cells_epochs_nodes(
            nodes,
            epochs,
            visualize,
            stats,
            output_dir,
            checkpoint_interval,
            radius,
            score_keys,
            solution_concept,
            **kwargs
        )
    elif cell_evaluation == "all_vs_all":
        nodes = evaluate_cells_all_vs_all(
            nodes,
            epochs,
            visualize,
            stats,
            output_dir,
            checkpoint_interval,
            score_keys,
            solution_concept,
            **kwargs
        )
    elif cell_evaluation == "ann_canonical":
        assert population_size == 1
        nodes = evaluate_ann_canonical(
            nodes,
            epochs,
            visualize,
            stats,
            output_dir,
            checkpoint_interval,
            score_keys,
            **kwargs
        )
    elif cell_evaluation == "async":
        raise NotImplementedError("async is not implemented")

    best_encoder, best_decoder, idx = get_best_nodes(
        list(nodes.values()), test_data, visualize, epochs, output_dir
    )
    if visualize != "none":
        show_images(best_decoder, output_dir)

    plot_stats(stats, output_dir, id=kwargs["timestamp"])
    out_path = os.path.join(output_dir, "checkpoints", f"t{epochs}_ae.pt")
    data = {"encoder": best_encoder.state_dict(), "decoder": best_decoder.state_dict()}
    torch.save(data, out_path)
    logging.debug(f"Save best ANNs from {idx} to {out_path}")
    if kwargs.get("store_all_nodes", False):
        logging.info("Save all node centers")
        for node in nodes.values():
            ensemble_path = os.path.join(output_dir, "checkpoints", "ensemble")
            os.makedirs(ensemble_path, exist_ok=True)
            out_path = os.path.join(ensemble_path, f"n{node.index}_t{epochs}_ae.pt")
            data = {"encoder": node.encoders[0].state_dict(), "decoder": node.decoders[0].state_dict()}
            torch.save(data, out_path)
        

    return best_encoder, best_decoder


def get_best_nodes(
    nodes: List[Node],
    test_data: DataLoader,
    visualize: str,
    iteration: int,
    output_dir: str,
) -> Tuple[torch.nn.Module, torch.nn.Module, int]:
    encoders = [node.encoders[0] for node in nodes]
    decoders = [node.decoders[0] for node in nodes]
    all_losses = np.zeros((len(encoders), 1, 1))
    for data, _ in test_data:
        data = data.to(device)
        for i, (e, d) in enumerate(zip(encoders, decoders)):
            # TODO assumes all nodes have the same autoencoder class
            ae = nodes[0].Autoencoder(e, d)
            loss = ae.compute_loss_together(data.view(-1, ae.x_dim))
            all_losses[i, 0, 0] = all_losses[i, 0, 0] + loss.data.item()

    e_sorted_idx = np.argsort(all_losses)
    idx = e_sorted_idx[0, 0, 0]
    best_encoder = encoders[idx]
    best_decoder = decoders[idx]
    if visualize == "all":
        plot_node_losses(all_losses, idx, iteration, "best", output_dir)
    if visualize != "none":
        plot_ae_data(ae, data, "test", output_dir)

    logging.info(f"Get best ANNs with losses {all_losses}")
    return best_encoder, best_decoder, idx


def calculate_losses(
    encoders: List[torch.nn.Module],
    decoders: List[torch.nn.Module],
    data: torch.Tensor,
    Autoencoder: torch.nn.Module,
) -> np.ndarray:
    # Sum loss, encoder loss, decoder loss
    all_losses = np.zeros((len(encoders), len(decoders), 3))
    for i, encoder in enumerate(encoders):
        for j, decoder in enumerate(decoders):
            ae = Autoencoder(encoder, decoder)
            ae = ae.to(device)
            loss = ae.compute_loss_together(data)
            all_losses[i, j, 0] = loss
            all_losses[i, j, 1] = ae.encoder.loss
            all_losses[i, j, 2] = ae.decoder.loss

    return all_losses


def worst_case(all_losses: np.array) -> np.array:
    return np.max(all_losses[:, :, 0], axis=1)


def best_case(all_losses: np.array) -> np.array:
    return np.min(all_losses[:, :, 0], axis=1)


def mean_expected_utility(all_losses: np.array) -> np.array:
    return np.mean(all_losses[:, :, 0], axis=1)


def evaluate_anns(
    encoders: List[torch.nn.Module],
    decoders: List[torch.nn.Module],
    input_data: DataLoader,
    Autoencoder: torch.nn.Module,
    solution_concept: callable
) -> Tuple[np.ndarray, np.ndarray]:
    # Evaluates on one batch
    # TODO assumes all encoders the same
    data = next(iter(input_data))[0].view(-1, encoders[0].x_dim)
    data = data.to(device)
    all_losses = calculate_losses(encoders, decoders, data, Autoencoder)
    # Get the worst case
    # TODO need both encoder and decoder loss
    losses = solution_concept(all_losses)
    logging.debug(
        f"Evaluated on one batch with {solution_concept} losses {all_losses.shape} {losses.shape} max losses: {losses} (min loss for all {np.min(all_losses[:, :, 0], axis=1)})"
    )
    return losses, all_losses


def update_ann(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    batch: torch.Tensor,
    learning_rate: float,
    Autoencoder: torch.nn.Module,
    optimizer: Any=None,
    **kwargs
) -> torch.nn.Module:
    logging.debug(f"BEGIN Update ANN {encoder.__class__.__name__}")
    ae = Autoencoder(encoder, decoder)
    ae = ae.to(device)
    ae.train()
    # TODO is optimizer used correctly. E.g. should we save the optimizer state, even though the encoder and decoder can be different
    optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            
    optimizer.zero_grad()
    # TODO MNIST hardcoding
    loss = ae.compute_loss_together(batch.view(-1, ae.x_dim))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ae.parameters(), sys.float_info.max / 1.0e10)
    optimizer.step()
    
    return encoder, decoder, optimizer


def select(
    node: Node, losses: np.ndarray
) -> Tuple[List[torch.nn.Module], List[torch.nn.Module]]:
    encoders = []
    decoders = []
    tournament_size = 2
    sub_population_size = 1
    for _ in range(sub_population_size):
        idxs = rng.integers(0, losses.shape[0], size=(tournament_size, 1))
        # TODO tournament size other than 2
        assert idxs.shape == (2, 1)
        # Check loss for the selected individuals
        if losses[idxs[0, 0]] < losses[idxs[1, 0]]:
            idx = idxs[0, 0]
        else:
            idx = idxs[1, 0]

        encoders.append(node.encoders[idx].clone())
        decoders.append(node.decoders[idx].clone())
    
    logging.debug(f"Selected {len(encoders)} from node {node.index}.")
    return encoders, decoders


def replace(
    node: Node,
    encoders: List[torch.nn.Module],
    decoders: List[torch.nn.Module],
    losses: np.ndarray,
) -> Node:
    e_sorted_idx = np.argsort(losses)
    node.encoders = [encoders[_] for _ in e_sorted_idx]
    d_sorted_idx = np.argsort(losses)
    node.decoders = [decoders[_] for _ in d_sorted_idx]
    logging.debug(
        f"Replace anns with ({(e_sorted_idx, d_sorted_idx)}) in node {node.index}"
    )
    return node


def evaluate_cell(
    node,
    learning_rate: float,
    iteration: int,
    visualize: str,
    stats: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_interval: int,
    score_keys: List[str],
    last_iteration: int,
    solution_concept: callable,
    **kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    logging.debug(
        f"Evaluate node {node.index} with learning rate {learning_rate} at iteration {iteration}"
    )
    training_data = node.training_data
    overwrite_checkpoint = kwargs.get("overwrite_checkpoint", True)
    stat = {
        "iteration": iteration,
        "learning_rate": learning_rate,
        "node_idx": node.index,        
    }
    if len(stats) > 0 and stat["iteration"] == stats[-1]["iteration"]:
        assert (stat["node_idx"] != stats[-1]["node_idx"]), f"{stat} {stats[-1]} {len(stats)}\n{stats}" 

    # Get ANNs
    E, D = (node.encoders, node.decoders)
    # Get non evolving ANN
    # Get Random Batch
    # Evaluate
    losses, all_losses = evaluate_anns(E, D, training_data, node.Autoencoder, solution_concept)
    stat["min_selection_loss"] = np.min(losses)
    logging.debug(f"Min selection loss: {stat['min_selection_loss']} for {node.index}")
    if visualize == "all":
        plot_node_losses(all_losses, node.index, iteration, "select", output_dir)
    # Select
    E_p, D_p = select(node, losses)
    for i, (batch, _) in enumerate(training_data):
        logging.debug(f"Batch {i}")
        batch = batch.to(device)
        # Get random opponent
        d_p = get_opponent(D)
        for e in E_p:
            # Update ANN
            e, _, o = update_ann(e, d_p, batch, learning_rate, node.Autoencoder, node.optimizer, **kwargs)
            node.optimizer = o
        # Get random opponent
        e_p = get_opponent(E)
        for d in D_p:
            # Update ANN
            _, d, o = update_ann(e_p, d, batch, learning_rate, node.Autoencoder, node.optimizer, **kwargs)
            node.optimizer = o

    # Evaluate
    losses, all_losses = evaluate_anns(E_p, D_p, training_data, node.Autoencoder, solution_concept)
    if visualize == "all":
        plot_node_losses(all_losses, node.index, iteration, "replace", output_dir)

    # Replace and set center
    node = replace(node, E_p, D_p, losses)
    stat["min_replacement_loss"] = np.min(losses)
    stat["timestamp"] = float(datetime.datetime.timestamp(datetime.datetime.now()))
    if kwargs.get("calculate_test_loss", False):
        test_data = kwargs.get("test_data", None)
        E, D = node.encoders[:1], node.decoders[:1]
        losses, all_losses = evaluate_anns(E, D, test_data, node.Autoencoder, solution_concept)
        stat["test_loss"] = np.min(losses)
        logging.info(f"Test loss: {stat['test_loss']}")
        
    fe_cnt = 0 if len(stats) == 0 else stats[-1]["fe_cnt"]
    stats.append(stat)
    logging.info(f"Epoch {iteration} Min selection loss: {stats[-1]['min_selection_loss']} fe_cnt: {fe_cnt}")
    if iteration % loss_report_interval == 0 or\
            iteration == last_iteration:
        ae = node.Autoencoder(node.encoders[0], node.decoders[0])
        logging.debug(f"{iteration} calculating quality scores on training_data")                    
        score_values = measure_quality_scores(score_keys, ae, training_data) 
        stat.update(score_values)
            
        if visualize == "all":
            plot_train_data(node, batch, iteration, node.Autoencoder, output_dir)

        logging.debug(f"Min losses after training: {np.min(losses)}")

    if iteration % checkpoint_interval == 0:
        # NOTE overwrites checkpoints
        out_path = os.path.join(output_dir, "checkpoints", f"n{node.index}_ae.pt")
        if not overwrite_checkpoint:
            out_path = os.path.join(output_dir, "checkpoints", f"t_{iteration}_n{node.index}_ae.pt")
        data = {
            "encoder": node.encoders[0].state_dict(),
            "decoder": node.decoders[0].state_dict(),
        }
        torch.save(data, out_path)
        logging.debug(f"Checkpoint ANNs for {node.index} in {out_path}")

    return node


if __name__ == "__main__":
    _timestamp = datetime.datetime.now()
    timestamp_str = '{:%Y-%m-%d_%H:%M:%S.%f}'.format(_timestamp)    
    log_file = os.path.basename(__file__).replace(".py", f"_{timestamp_str}.log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    param = parse_arguments(sys.argv[1:])
    logging.getLogger().setLevel(level=param.log_level.upper())
    print(logging.getLogger().level, param.log_level)
    # TODO messy, how about overriding from the command line?
    if param.configuration_file:
        logging.info(f"Loading param from file {param.configuration_file} all specified command-line arguments are ignored")
        with open(param.configuration_file, "r") as fd:
            params = json.load(fd)        
    else:
        params = vars(param)

    params["timestamp"] = datetime.datetime.timestamp(_timestamp)
    params["output_dir"] = f'{params["output_dir"]}_{timestamp_str}'
    logging.info(f"Using params {params}")
    # TODO ugly
    os.makedirs(params["output_dir"], exist_ok=True)
    out_path = os.path.join(params["output_dir"], "params.json")
    with open(out_path, "w") as fd:
        json.dump(params, fd, indent=1)

    if param.no_execution:
        logging.info("No execution flag")
        sys.exit(0)
    
    try:  
        main(**params)
    except RuntimeError as e:
        logging.error(e)        
