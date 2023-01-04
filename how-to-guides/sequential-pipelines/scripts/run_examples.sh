set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running data_preprocessing.py..."
python data_preprocessing.py

echo "Running model_training.py"

echo "Running training.py"
