set -e

echo "Installing requirements..."
pip install -q -U -r requirements.txt

echo "Running re_run_failed_training.py..."
python re_run_failed_training.py
