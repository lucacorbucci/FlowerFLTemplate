# FLTemplate

FLTemplate is a flexible federated learning framework built on top of [Flower](https://flower.ai/) (FLwr). It enables simulation of federated learning experiments in both cross-device and cross-silo settings, supporting IID and non-IID data partitioning (using Dirichlet distribution). The framework is designed for ease of use, with support for various datasets, classification and regression tasks, and integration with Weights & Biases (WandB) for experiment tracking and hyperparameter sweeps.

## Features

- **Federated Learning Simulations**: Implements FedAvg strategy for distributed model training.
- **Dataset Support**: Built-in support for:
  - MNIST (classification, image data)
  - Abalone (regression, tabular data)
  - Dutch (classification, tabular with sensitive attributes)
  - Income/Adult (classification, tabular with sensitive attributes)
- **Partitioning**: IID (uniform) and non-IID (Dirichlet-based) data partitioning.
- **Settings**: Cross-device (simulated clients) and cross-silo (realistic node setups).
- **Hyperparameter Sweeps**: Support for WandB sweeps.
- **Metrics and Logging**: Aggregates losses, accuracy (classification), RMSE/MAE/R2 (regression), and logs to WandB.
- **Extensibility**: Modular structure for custom models, strategies, and datasets.

## Installation



## Usage

FLTemplate uses YAML configuration files for experiments. Run simulations using the main script with command-line arguments.

### Basic Run

To run a simulation for MNIST in cross-silo IID setting:


### Configuration

Experiments are configured via command-line args or YAML files (see examples/ directory).

Key parameters:
- `--num_clients`: Number of clients.
- `--num_rounds`: Number of FL rounds.
- `--FL_setting`: "cross_device" or "cross_silo".
- `--dataset_name`: "mnist", "abalone", "dutch", "income".
- `--partitioner_type`: "iid" or "non_iid" (with `--partitioner_alpha` for Dirichlet alpha).
- `--sampled_training_nodes_per_round`: Fraction of clients for training per round.
- `--wandb`: Enable WandB logging.

For hyperparameter sweeps, use `--sweep true` and run `run_sweep.sh` in examples/.

### Examples

The `examples/` directory contains YAML configs and scripts for each dataset:

- **MNIST Cross-Silo IID**: `examples/mnist/mnist_cross_silo_iid.yaml`
  ```
  bash examples/mnist/run.sh
  ```

- **Abalone Cross-Silo**: `examples/abalone/abalone_cross_silo_iid.yaml`
  ```
  bash examples/abalone/run_sweep.sh
  ```

Similar scripts for Dutch and Income datasets with IID/non-IID variants.

### File Structure

```
FLTemplate/
├── main.py                 # Entry point for simulations
├── Aggregations/           # Metric aggregation functions
│   └── aggregations.py
├── Client/                 # Client implementation
│   └── client.py
├── ClientManager/          # Client management
│   └── client_manager.py
├── Datasets/               # Dataset loaders and preprocessors
│   ├── abalone.py
│   ├── dutch.py
│   ├── income.py
│   ├── mnist.py
│   └── dataset_utils.py
├── Models/                 # Model architectures and wrappers
│   ├── architectures.py   # Specific models (e.g., AbaloneNet)
│   ├── model.py           # Abstract Model base class
│   ├── regression_model.py # Regression wrapper
│   ├── simple_model.py    # Classification wrapper
│   └── utils.py           # Model utilities (get_model)
├── Server/                 # Server implementation
│   └── server.py
├── Strategy/               # FL strategies
│   └── fed_avg.py
└── Utils/                  # Utilities
    ├── preferences.py      # Configuration dataclass
    └── utils.py            # PyTorch utilities (set/get params, optimizer)
```

## Datasets

- **MNIST**: Image classification (10 classes). Downloaded to `data/MNIST/train/<class>/<image>.png`.
- **Abalone**: Regression (predict rings/age from physical measurements). CSV format, scaled with StandardScaler.
- **Dutch**: Classification (occupation binary), sensitive attribute (sex). Tabular CSV, MinMaxScaler.
- **Income (Adult)**: Classification (>50K income), sensitive attribute (sex). Multiple CSVs concatenated, TargetEncoder for categoricals, StandardScaler for continuous.

Data is partitioned per client, with train/test splits.

### How to add a new Dataset

# TODO

If you already have the dataset on you machine, you can add it by following these steps:

- In the main.py file you have to add the new dataset to the prepare_data function. 

```python
    elif preferences.dataset_name == "speech_fairness":
        data_info = get_data_info(preferences)
        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
```

This holds in the case of a csv dataset, for images for instance you can use

```python
    elif preferences.dataset_name == "your_dataset_name":
        data_info = get_data_info(preferences)
        dataset_dict = load_dataset("imagefolder", data_dir=preferences.dataset_path)
```



## Running Experiments

1. Prepare data (for MNIST, run download in mnist.py if needed).
2. Configure YAML or args.
3. Run `python FLTemplate/main.py [args]`.
4. View logs and WandB dashboard for metrics.

For sweeps, edit YAML and run `run_sweep.sh`.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Flower](https://flower.ai/).
- Datasets from UCI and Hugging Face.
- Logging with Weights & Biases.
