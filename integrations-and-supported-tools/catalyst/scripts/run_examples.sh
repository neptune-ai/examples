set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Catalyst.py..."
python Neptune_Catalyst.py
