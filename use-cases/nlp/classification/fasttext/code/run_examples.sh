set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running fasttext_script.py..."
python fasttext_script.py
