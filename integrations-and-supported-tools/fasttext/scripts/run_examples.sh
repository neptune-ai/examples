set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_fastText.py..."
python Neptune_fastText.py
