set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Matplotlib_Support.py..."
python Neptune_Matplotlib_Support.py
