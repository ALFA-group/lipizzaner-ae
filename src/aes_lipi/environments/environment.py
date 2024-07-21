import argparse
import logging
import os
import sys
from typing import List


# TODO refactor train and test here?


def parse_arguments(param: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments (`sys.argv`).
    """
    parser = argparse.ArgumentParser(description="Run Environment")
    parser.add_argument(
        "--method",
        type=str,
        help="Autoencoder method. E.g. DenoisingAutoencoder",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name. E.g. mnist",
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
    # create_binary_cluster_data()
    params = parse_arguments(sys.argv[1:])
    print(params)
    # TODO write test for running all methods
