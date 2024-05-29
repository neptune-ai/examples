set -e

echo "Installing requirements..."
pip install -q -U -r requirements.txt

echo "Running Neptune_Pytorch_Lightning.py..."
python Neptune_Pytorch_Lightning.py
