set -e

echo "Installing requirements..."
pip install -q -U -r requirements.txt

echo "Running Neptune_MosaicML_Composer.py..."
python Neptune_MosaicML_Composer.py
