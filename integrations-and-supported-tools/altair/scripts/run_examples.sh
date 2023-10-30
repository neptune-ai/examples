set -e

echo "Installing requirements..."
pip install -U -q -r requirements.txt

echo "Running Neptune_Altair_Support.py..."
python Neptune_Altair_Support.py
