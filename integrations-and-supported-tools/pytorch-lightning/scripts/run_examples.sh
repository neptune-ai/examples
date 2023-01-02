set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Pytorch_Support.py..."
python Neptune_Pytorch_Lightning.py

echo "Running Neptune_Pytorch_Support_more_options.py..."
python Neptune_Pytorch_Lightning_more_options.py
