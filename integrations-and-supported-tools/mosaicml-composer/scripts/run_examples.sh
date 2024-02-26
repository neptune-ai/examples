set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_MosaicML_Composer.py..."
python Neptune_MosaicML_Composer.py
