set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running neptune_cross_valition.py ..."
python neptune_cross_validation.py
