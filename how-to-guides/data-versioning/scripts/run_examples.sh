set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Version_datasets_in_model_training_runs.py..."
python Version_datasets_in_model_training_runs.py

echo "Running Compare_model_training_runs_on_dataset_versions.py..."
python Compare_model_training_runs_on_dataset_versions.py

echo "Running Organize_and_share_dataset_versions.py..."
python Organize_and_share_dataset_versions.py
