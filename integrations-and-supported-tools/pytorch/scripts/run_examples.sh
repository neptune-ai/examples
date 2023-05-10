set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Pytorch.py..."
python Neptune_Pytorch.py
