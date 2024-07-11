set -e

echo "Installing requirements..."
pip install -q -U -r requirements.txt

echo "Running Neptune_Pytorch_Ignite.py..."
python Neptune_Pytorch_Ignite.py
