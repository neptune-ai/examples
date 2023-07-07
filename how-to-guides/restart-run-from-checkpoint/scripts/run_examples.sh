set -e

export NEPTUNE_CUSTOM_RUN_ID=`date +"%Y%m%d%H%M%s%N"`

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running neptune_save_checkpoints.py..."
python neptune_save_checkpoints.py

echo "Running neptune_restart_run_from_checkpoint.py..."
python neptune_restart_run_from_checkpoint.py
