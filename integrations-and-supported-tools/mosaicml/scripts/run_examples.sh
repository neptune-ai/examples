set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_MosaicML.py..."
python Neptune_MosaicML.py
