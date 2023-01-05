set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running hello_neptune.py..."
python hello_neptune.py
