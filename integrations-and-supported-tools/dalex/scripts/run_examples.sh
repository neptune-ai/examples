set -e

echo "Installing requirements..."
pip install -U --user -r requirements.txt

echo "Running Neptune_Dalex.py..."
python Neptune_Dalex.py
