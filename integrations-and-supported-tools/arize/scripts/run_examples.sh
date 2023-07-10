set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Arize.py..."
python Neptune_Arize.py
