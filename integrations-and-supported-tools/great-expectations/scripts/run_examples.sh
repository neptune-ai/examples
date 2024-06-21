set -e

echo "Installing requirements..."
pip install -q -U -r requirements.txt

echo "Running Neptune_Great_Expectations.py..."
python Neptune_Great_Expectations.py
