# FLTemplate

FLTemplate is a flexible federated learning framework built on top of [Flower](https://flower.ai/) (FLwr). It enables simulation of federated learning experiments in both cross-device and cross-silo settings, supporting IID and non-IID data partitioning (using Dirichlet distribution). The framework is designed for ease of use, with support for various datasets, classification and regression tasks, and integration with Weights & Biases (WandB) for experiment tracking and hyperparameter sweeps.

## Features

- **Federated Learning Simulations**: Implements FedAvg strategy for distributed model training.
- **Dataset Support**: Built-in support for:
  - MNIST (classification, image data). 
  - Abalone (regression, tabular data). This is available in /data/abalone as a CSV file.
  - Dutch (classification, tabular with sensitive attributes). This is available in /data/dutch as a CSV file.
  - ACS Income (classification, tabular with sensitive attributes). For this dataset, in the /data/income_reduced folder you can find a smaller version of the [original one](https://arxiv.org/abs/2108.04884).
- **Partitioning**: IID (uniform) and non-IID (Dirichlet-based) data partitioning.
- **Settings**: Cross-device (simulated clients) and cross-silo (realistic node setups).
- **Hyperparameter Sweeps**: Support for WandB sweeps.
- **Metrics and Logging**: Aggregates losses, accuracy (classification), RMSE/MAE/R2 (regression), and logs to WandB.
- **Extensibility**: Modular structure for custom models, strategies, and datasets.

## Before You Start

This project uses uv to manage dependencies. Make sure you have it installed. You can install it via pip:

```bash
pipx install uv
```

or using curl:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or wget:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Fore more details, visit the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Basic Usage

If you want to try out a simple simulation with default settings, you can run:

```bash
cd /FLTemplate/examples/dutch/
uv run python /home/lcorbucci/FLTemplate/FLTemplate/examples/dutch/../../main.py --batch_size=51 --lr=0.027523254839401178 --momentum=0.037879525096583266 --num_epochs=3 --optimizer=adam --weight_decay=0.0009210304960670968 --dataset_name dutch --num_rounds 10 --num_clients 20 --FL_setting cross_device --sampled_train_nodes_per_round 1 --sampled_validation_nodes_per_round 1 --sampled_test_nodes_per_round 0 --fed_dir ../../../training_data/dutch/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../../../data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1 --partitioner_by occupation --num_train_nodes 12 --num_validation_nodes 4 --num_test_nodes 4 --sweep True
```

This will run a federated learning simulation on the Dutch dataset with specified hyperparameters.
Make sure you have the dataset in the specified path and you have set up WandB if you want to log the results.

## Configuration

Experiments are configured via command-line args or YAML files (during hyperparameter tuning).

Key parameters:
- `--num_clients`: Number of clients involved in the simulation.
- `--num_rounds`: Number of FL rounds that will be performed.
- `--num_epochs`: Local epochs per client.
- `--batch_size`: Local batch size for training.
- `--lr`: Learning rate.
- `--optimizer`: Optimizer type ("sgd", "adam").
- `--momentum`: Momentum for SGD.
- `--weight_decay`: Weight decay (L2 regularization).
- `--FL_setting`: "cross_device" or "cross_silo".
- `--dataset_name`: "mnist", "abalone", "dutch", "income".
- `--partitioner_type`: "iid" or "non_iid" (with `--partitioner_alpha` for Dirichlet alpha).
- `--partitioner_alpha`: Dirichlet alpha parameter (float).
- `--partitioner_by`: Attribute to partition by (str).
- `--num_train_nodes`: Number of training nodes/clients. This is used in the cross-device setting.
- `--num_validation_nodes`: Number of validation nodes/clients. This is used in the cross-device setting.
- `--num_test_nodes`: Number of test nodes/clients. This is used in the cross-device setting.
- `--sampled_train_nodes_per_round`: Fraction of clients for training per round. E.g., 0.1 means 10% of clients.
- `--sampled_validation_nodes_per_round`: Fraction of clients for validation per round.
- `--sampled_test_nodes_per_round`: Fraction of clients for testing per round.
- `--fed_dir`: Directory where the results, logs and files related to the federated learning experiment will be saved.
- `--dataset_path`: Path to the dataset file (CSV for tabular, folder for images).
- `--sweep`: Enable hyperparameter sweep
- `--wandb`: Enable WandB logging
- `--project_name`: WandB project name.
- `--run_name`: WandB run name.
- `--task`: "classification" or "regression".
- `--image_path`: Path to the folder containing the images (for MNIST).

## Run an Hyperparameter Sweep

You can see an example of how to run a sweep in the `examples/` directory. Each dataset has its own folder with YAML configs and scripts.
For instance, to run a sweep on the MNIST dataset in a cross-silo IID setting we have to use the following YAML file: 

```yaml
program: ../../main.py
method: bayes
metric:
  name: Validation_Accuracy
  goal: maximize
early_terminate:
  type: hyperband
  min_iter: 3
parameters:
  batch_size:
    min: 32
    max: 512
  lr:
    min: 0.0001
    max: 0.1
  optimizer:
    values: ["adam", "sgd"]
  momentum:
    min: 0.0
    max: 0.9
  weight_decay:
    min: 0.00001
    max: 0.001
  num_epochs:
    min: 1
    max: 3

command:
  - ${env}
  - uv 
  - run 
  - python
  - ${program}
  - ${args}
  - --dataset_name
  - mnist
  - --num_rounds
  - 10
  - --num_clients
  - 20
  - --FL_setting 
  - cross_silo
  - --sampled_train_nodes_per_round
  - 1.0
  - --sampled_validation_nodes_per_round
  - 1.0
  - --sampled_test_nodes_per_round
  - 0.0
  - --fed_dir 
  - ../../../training_data/mnist/ 
  - --project_name 
  - TestTemplateFL 
  - --run_name 
  - Test 
  - --wandb 
  - True 
  - --dataset_path 
  - ../../../data/MNIST/train/
  - --partitioner_type 
  - iid 
  - --sweep
  - True
```

To start the sweep, you can use the script run_sweep.sh provided in the same folder.

### How to add a new Dataset

TODO


## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Make sure the code you pushed passes linting tests. 
6. Open a Pull Request.


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Flower](https://flower.ai/).
- Datasets from UCI and Hugging Face.
- Logging with Weights & Biases.