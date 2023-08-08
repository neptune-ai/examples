set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Pandas.py..."
python Neptune_Pandas.py
