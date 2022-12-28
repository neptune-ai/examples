set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_prophet.pys..."
python Neptune_prophet.py

echo "Running Neptune_prophet_more_options.pys..."
python Neptune_prophet_more_options.py
