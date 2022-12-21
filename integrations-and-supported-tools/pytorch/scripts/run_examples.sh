set -e

echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Neptune_Pytorch_Support.py..."
python Neptune_Pytorch_Support.py

echo "Running Neptune_Pytorch_Support_more_options.py..."
python Neptune_Pytorch_Support_more_options.py
