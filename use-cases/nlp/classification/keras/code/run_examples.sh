set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running keras_script.py..."
python keras_script.py
