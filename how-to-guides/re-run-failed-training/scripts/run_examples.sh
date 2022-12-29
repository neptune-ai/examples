set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running re_run_failed_training.py..."
python re_run_failed_training.py
