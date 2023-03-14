set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Checkout to commit ID 42ff89ed836b8240cce582eec7e7840bc412ebfd ..."
git checkout 42ff89ed836b8240cce582eec7e7840bc412ebfd

echo "Running old_run.py..."
python old_run.py
