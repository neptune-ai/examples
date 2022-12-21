set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Sacred.py..."
python Neptune_Sacred.py

echo "Running Neptune_Sacred_more_options.py..."
python Neptune_Sacred_more_options.py
