import torch
from Dataset.federated_dataset import FederatedDataset
from Dataset.load_data import LoadDataset
from Dataset.tabular_dataset import TabularDataset
from Utils.preferences import Preferences

# # iid test
# preferences = Preferences(
#     dataset="adult",
#     dataset_path="../data/adult/",
#     epochs=10,
#     device="cpu",
#     batch_size=64,
#     seed=42,
#     optimizer="adam",
#     sweep=False,
#     fl_round=1,
#     cross_device=False,
#     tabular = True,
#     num_nodes = 10,
#     split_approach="iid",
# )


# if preferences.tabular:
#     if preferences.cross_device:
#         X, Z, y = LoadDataset.load_dataset(preferences)
#     else:
#         X, Z, y, X_test, Z_test, y_test = LoadDataset.load_dataset(preferences)
#         FederatedDataset.create_federated_dataset(preferences, X, y, Z, "train")
#         FederatedDataset.create_federated_dataset(preferences, X_test, y_test, Z_test, "test")

# non iid test
# preferences = Preferences(
#     dataset="adult",
#     dataset_path="../data/adult/",
#     epochs=10,
#     device="cpu",
#     batch_size=64,
#     seed=42,
#     optimizer="adam",
#     sweep=False,
#     fl_round=1,
#     cross_device=False,
#     tabular = True,
#     num_nodes = 10,
#     split_approach="non_iid",
#     alpha_dirichlet=1.0
# )


# if preferences.tabular:
#     if preferences.cross_device:
#         X, Z, y = LoadDataset.load_dataset(preferences)
#     else:
#         X, Z, y, X_test, Z_test, y_test = LoadDataset.load_dataset(preferences)
#         FederatedDataset.create_federated_dataset(preferences, X, y, Z, "train")
#         FederatedDataset.create_federated_dataset(preferences, X_test, y_test, Z_test, "test")

# for node in range(preferences.num_nodes):
#     print(f"Node {node}:")
#     print(torch.load(f"../data/adult/federated/{node}/train.pt").targets)
#     print(torch.load(f"../data/adult/federated/{node}/test.pt").targets)


# representative

preferences = Preferences(
    dataset="adult",
    dataset_path="../data/adult/",
    epochs=10,
    device="cpu",
    batch_size=64,
    seed=42,
    optimizer="adam",
    sweep=False,
    fl_round=1,
    cross_device=False,
    tabular=True,
    num_nodes=10,
    split_approach="representative_diversity",
    ratio_unfair_nodes=0.3,
    group_to_reduce=(0.0, 1), 
    group_to_increment=(1.0, 1),
    ratio_unfairness=(0.1, 0.2)
)


if preferences.tabular:
    if preferences.cross_device:
        X, Z, y = LoadDataset.load_dataset(preferences)
    else:
        X, Z, y, X_test, Z_test, y_test = LoadDataset.load_dataset(preferences)
        FederatedDataset.create_federated_dataset(preferences, X, y, Z, "train")
        FederatedDataset.create_federated_dataset(
            preferences, X_test, y_test, Z_test, "test"
        )

for node in range(preferences.num_nodes):
    print(f"Node {node}:")
    print(torch.load(f"../data/adult/federated/{node}/train.pt").targets)
    print(torch.load(f"../data/adult/federated/{node}/test.pt").targets)
