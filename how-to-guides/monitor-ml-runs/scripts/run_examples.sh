set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Monitor_ML_runs_live.py..."
python Monitor_ML_runs_live.py
