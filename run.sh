# mnist cross-silo NON IID ✅
uv run python main.py --FL_setting cross_silo --num_clients 10 --dataset_name mnist --num_rounds 4 --sampled_train_nodes_per_round 1.0 --sampled_test_nodes_per_round 1.0 --fed_dir ../training_data/mnist/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1000 --partitioner_by label

# mnist cross-device NON IID ✅
uv run python main.py --FL_setting cross_device --num_clients 10 --dataset_name mnist --num_rounds 4 --sampled_train_nodes_per_round 1.0 --sampled_test_nodes_per_round 1.0 --fed_dir ../training_data/mnist/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1000 --partitioner_by label --num_train_nodes 7 --num_test_nodes 3


# mnist cross-silo IID ✅
uv run python main.py --FL_setting cross_silo --num_clients 10 --dataset_name mnist --num_rounds 4 --sampled_train_nodes_per_round 1.0 --sampled_test_nodes_per_round 1.0 --fed_dir ../training_data/mnist/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../data/MNIST/train/ --partitioner_type iid 

# mnist cross-device IID ✅
uv run python main.py --FL_setting cross_device --num_clients 10 --dataset_name mnist --num_rounds 4 --sampled_train_nodes_per_round 1.0 --sampled_test_nodes_per_round 1.0 --fed_dir ../training_data/mnist/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../data/MNIST/train/ --partitioner_type iid --num_train_nodes 7 --num_test_nodes 3



# mnist sweep test cross_device IID ✅
uv run python main.py --FL_setting cross_device --num_clients 10 --dataset_name mnist --num_rounds 5 --sampled_train_nodes_per_round 1.0 --sampled_validation_nodes_per_round 1.0 --sampled_test_nodes_per_round 0 --fed_dir ../training_data/mnist/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../data/MNIST/train/ --partitioner_type iid --num_train_nodes 4 --num_validation_nodes 3 --num_test_nodes 3 --sweep True

# mnist sweep test cross_silo NON IID 
uv run python main.py --FL_setting cross_silo --num_clients 10 --dataset_name mnist --num_rounds 5 --sampled_train_nodes_per_round 1.0 --sampled_validation_nodes_per_round 1.0 --sampled_test_nodes_per_round 0 --fed_dir ../training_data/mnist/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by label --sweep True


# dutch cross-silo NON IID ✅
uv run python main.py --FL_setting cross_silo --num_clients 10 --dataset_name dutch --num_rounds 4 --sampled_train_nodes_per_round 1.0 --sampled_test_nodes_per_round 1.0 --fed_dir ../training_data/dutch/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation

# dutch cross-device NON IID ✅
uv run python main.py --FL_setting cross_device --num_clients 10 --dataset_name dutch --num_rounds 4 --sampled_train_nodes_per_round 1.0 --sampled_test_nodes_per_round 1.0 --fed_dir ../training_data/dutch/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --num_train_nodes 7 --num_test_nodes 3

# dutch cross-silo IID ✅
uv run python main.py --FL_setting cross_silo --num_clients 10 --dataset_name dutch --num_rounds 4 --sampled_train_nodes_per_round 1.0 --sampled_test_nodes_per_round 1.0 --fed_dir ../training_data/dutch/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../data/dutch/dutch.csv --partitioner_type iid 

# dutch cross-device IID ✅ 
uv run python main.py --FL_setting cross_device --num_clients 10 --dataset_name dutch --num_rounds 4 --sampled_train_nodes_per_round 1.0 --sampled_test_nodes_per_round 1.0 --fed_dir ../training_data/dutch/ --project_name TestTemplateFL --run_name Test --wandb True --dataset_path ../data/dutch/dutch.csv --partitioner_type iid --num_train_nodes 7 --num_test_nodes 3
