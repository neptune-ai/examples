set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running reproduce_run.py..."
python reproduce_run.py

echo "Running run_git_checkout.sh ..."
sh run_git_checkout.sh
