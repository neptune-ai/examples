echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Version_datasets_in_model_training_runs.py..."
python Version_datasets_in_model_training_runs.py

echo "Running Compare_model_training_runs_on_dataset_versions.py..."
python Compare_model_training_runs_on_dataset_versions.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "