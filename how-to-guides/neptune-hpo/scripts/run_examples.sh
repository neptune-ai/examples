set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_hpo_single_run.py..."
python Neptune_hpo_single_run.py

echo "Running Neptune_hpo_separate_runs.py..."
python Neptune_hpo_separate_runs.py
