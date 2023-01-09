set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_HTML_Support.py..."
python Neptune_HTML_Support.py
