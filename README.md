# Lipizzaner Autoencoder (Lipi-Ae)

## Installation

Create virtual environment. E.g
```
python3 -m venv ~/.venvs/lipi_ae_gecco_24
```

Activate virtual environment. E.g
```
source ~/.venvs/lipi_ae_gecco_24/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

## Quick start

### Binary Clustering

Create Binary Clustering problems
```
PYTHONPATH=src python src/aes_lipi/datasets/data_loader.py --n_dim 1000 --n_clusters 10
```

Test binary clustering problem and autoencoder
```
PYTHONPATH=src python src/aes_lipi/environments/binary_clustering.py --method=Autoencoder --dataset_name=binary_clustering_10_100_1000
```

Run Lipi-Ae on binary clustering problem
```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --configuration_file=tests/gecco_2024/configurations/binary_clustering/test_bc/binary_clustering_epoch_node_demo_lipi_ae.json
```

#### Experiment

Run experiments
```
 time PYTHONPATH=src python src/aes_lipi/utilities/gecco_experiments.py --configuration_directory tests/gecco_2024/configurations/binary_clustering/test_bc --sensitivity tests/gecco_2024/configurations/binary_clustering/test_bc/sensitivity_values.json
```

Update dataset in `sensitivity_values.json` key `"dataset_name"` by adding the new dataset to the list

Analyze data from `--root_dir` based on `--param_dir` parameters.
```
time PYTHONPATH=src python src/aes_lipi/utilities/analyse_data.py --root_dir out_binary_clustering --param_dir out_binary_clustering 
```

##### Compare ANN parameters

Save the parameters at every iteration
Run Lipi-Ae with solution concept `best_case`
```
PYTHONPATH=src python src/aes_lipi/lipi_ae.py --dataset_name binary_clustering_10_100_1000  --environment AutoencoderBinaryClustering --epochs 3 --batch_size 400 --population_size 2 --ae_quality_measures L1 --solution_concept best_case --checkpoint_interval 1 --do_not_overwrite_checkpoint
```
## Reference

```

@inproceedings{hemberg2024ae,
  title={Cooperative Spatial Topologies for Autoencoder Training},
  author={Hemberg, Erik and Toutouh, Jamal and O'Reilly, Una-May},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  year={2024}
}
```