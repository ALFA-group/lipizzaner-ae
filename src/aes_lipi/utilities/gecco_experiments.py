import argparse
import datetime
import itertools
import logging
from math import ceil
import os
import json
import copy
import sys
from typing import List, Any, Dict

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from aes_lipi import lipi_ae
from aes_lipi.utilities.utilities import measure_ae_quality, str2bool


def main(configuration_file: str, sensitivity_file: str, timestamp: datetime="NA", create_config_files_only: bool=False, base_dir: str="experiments_tec") -> Dict[str, Any]:
    logging.info(f"Running experiment with {configuration_file} at {timestamp}")
    with open(configuration_file, "r") as fd:
        params = json.load(fd)

    params["timestamp"] = timestamp
    output_dir = params["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    variant_keys = ("population_size", "radius", "learning_rate")
    if sensitivity_file == "":
        values = dict([(k, [params[k]]) for k in variant_keys])               
    else:
        with open(sensitivity_file, 'r') as fd:
            values = json.load(fd)
        
    products = list(itertools.product(*values.values()))
    variants = []
    for product in products:
        variants.append(dict((k ,v) for k, v in zip(values.keys(), product)))
    
    single_variants = []
    for variant in variants:        
        trials = variant.get("trials", params["trials"])
        logging.info(f"variant: {variant} for {trials}")
        trial_params = copy.deepcopy(params)
        trial_params.update(variant)
        
        if is_blacklisted(trial_params, single_variants):
            continue
                            
        for i in range(trials):
            logging.info(f"Trial {i} for {configuration_file}")
            trial_params = copy.deepcopy(params)
            trial_params.update(variant)
            if trial_params.get("cell_evaluation", "") == "ann_canonical":
                trial_params["population_size"] = 1
            
            for k, v in trial_params.items():
                if v in ("True", "False"):
                    trial_params[k] = str2bool(v)
                    assert isinstance(trial_params[k], bool)                    
                    
            checkpoint_interval = trial_params["checkpoint_interval"]
            epochs = params.get("epochs")

            variant_name = '_'.join(['_'.join([str(k), str(v)]) for k, v in variant.items()])
            _time_stamp = datetime.datetime.now()
            trial_params["output_dir"] = os.path.join(output_dir, f"trial_{i}_{variant_name}")
            trial_params["rng_seed"] = params.get("rng_seed",  int(datetime.datetime.timestamp(_time_stamp))) + i 
            trial_params["epochs"] = epochs 
            trial_params["checkpoint_interval"] = checkpoint_interval
            if create_config_files_only: 
                _dir = os.path.join(base_dir, trial_params["output_dir"])
                os.makedirs(_dir, exist_ok=True)
                out_path = os.path.join(_dir, "params.json")    
                with open(out_path, "w") as fd:
                    json.dump(trial_params, fd, indent=1)
                logging.info(f"Not running. Only saving at {out_path}")
            else:
                timestamp_str = '{:%Y-%m-%d_%H:%M:%S.%f}'.format(_time_stamp)        
                trial_params["output_dir"] = f'{trial_params["output_dir"]}_{timestamp_str}'
                os.makedirs(trial_params["output_dir"], exist_ok=True)
                out_path = os.path.join(trial_params["output_dir"], "params.json")
                with open(out_path, "w") as fd:
                    json.dump(trial_params, fd, indent=1)

                try:
                    lipi_ae.main(**trial_params)
                except Exception as e:
                    logging.error(f"{e} PLEASE RERUN\n{trial_params}")
    return params


def is_blacklisted(variant: Dict[str, Any], single_variants: List[Dict[str, Any]]) -> bool:
    """ Check insensible variant"""
    cell_evaluation = variant.get("cell_evaluation", "")
    if cell_evaluation == "epoch-node":
        return False
    
    node_update_interval = variant.get("node_update_interval", "")
    radius = variant.get("radius", "")
    solution_concept = variant.get("solution_concept", "")
    population_size = variant.get("population_size", "")
    v_key = (node_update_interval, radius, solution_concept, population_size)
    for single_variant in single_variants:
        sv_key = tuple([single_variant[key] for key in ("node_update_interval", "radius", "solution_concept", "population_size")])
        if cell_evaluation == "ann_canonical" and single_variant["cell_evaluation"] == cell_evaluation:
            if v_key != sv_key:
                return True
        else:
            if v_key[:-2] != sv_key[:-2]:
                return True

    single_variants.append(variant)
    return False
    
    
def parse_arguments(param: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments (`sys.argv`).
    """
    parser = argparse.ArgumentParser(description="Run AE ANN experiment")
    parser.add_argument(
        "--configuration_file",
        type=str,
        default="configurations/binary_clustering_epoch_node_lipi_ae.json",
        help="JSON configuration file. E.g. " "configurations/demo_lipi_ae.json",
    )
    parser.add_argument(
        "--configuration_directory",
        type=str,
        help="Directory with JSON configuration file. E.g. "
        "configurations/experiments",
    )
    parser.add_argument(
        "--create_config_files_only",
        action="store_true",
        help="Create configuration files",
    )
    parser.add_argument(
        "--sensitivity",
        type=str,
        default="",
        help="Parameter sensitivity testing file, E.g. configurations/sensitivity_values.json",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["info", "debug", "warning"],
        default="info",
        help="Loglevel, e.g. --log_level info",
    )
    

    args = parser.parse_args(param)
    return args


def get_configuration_files(configuration_directory: str) -> List[str]:
    configuration_files = []
    for _file in os.listdir(configuration_directory):
        if not _file.endswith(".json") or\
            _file.startswith("sensitivity_values.json"):
            continue
        _file = os.path.join(configuration_directory, _file)
        configuration_files.append(_file)

    logging.info(f"Got {len(configuration_files)} from {configuration_directory}")
    return configuration_files


if __name__ == "__main__":
    timestamp = '{:%Y-%m-%d_%H:%M:%S.%f}'.format(datetime.datetime.now())
    log_file = os.path.basename(__file__).replace(".py", f"_{timestamp}.log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    param = parse_arguments(sys.argv[1:])
    logging.getLogger().setLevel(level=param.log_level.upper())
    print(logging.getLogger().level, param.log_level)
    if param.configuration_directory is not None:
        configuration_files = get_configuration_files(param.configuration_directory)
        for configuration_file in configuration_files:
            params = main(configuration_file, param.sensitivity, timestamp, param.create_config_files_only)
            if param.create_config_files_only:
                logging.info("Only creating configuration files only. Exiting")
                sys.exit(0)
                            
    else:
        main(param.configuration_file, param.sensitivity, timestamp, param.create_config_files_only)
