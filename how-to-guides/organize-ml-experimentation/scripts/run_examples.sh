set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Organize_ML_runs.py..."
python Organize_ML_runs.py
