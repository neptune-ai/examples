set -e

echo "Installing requirements..."
pip install --user -q -U -r requirements.txt --no-warn-script-location

echo "Running e2e_tracking.py..."
python e2e_tracking.py
