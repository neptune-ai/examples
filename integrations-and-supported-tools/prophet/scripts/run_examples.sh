set -e

echo "Installing requirements..."
pip install -U -r requirements.txt --user

echo "Running Neptune_prophet.py..."
python Neptune_prophet.py

echo "Running Neptune_prophet_more_options.py..."
python Neptune_prophet_more_options.py
