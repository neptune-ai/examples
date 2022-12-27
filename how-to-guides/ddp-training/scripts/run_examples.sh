set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running train_ddp_multiple_runs.py..."
torchrun --nproc_per_node=2 --nnodes=1 train_ddp_multiple_runs.py

echo "Running train_ddp_single_run.py..."
torchrun --nproc_per_node=2 --nnodes=1 train_ddp_single_run.py
