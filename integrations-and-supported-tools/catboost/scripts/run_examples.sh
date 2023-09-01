set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_CatBoost.py..."
python Neptune_CatBoost.py
