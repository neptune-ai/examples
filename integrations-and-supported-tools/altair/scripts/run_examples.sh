set -e

echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Neptune_Altair_Support.py..."
python Neptune_Altair_Support.py
