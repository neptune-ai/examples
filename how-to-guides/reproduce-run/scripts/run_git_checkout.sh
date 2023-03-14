set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Checkout to commit ID ee906dbc6a5f3f9f5cf06e49edb005cc2d6d2c75 ..."
git checkout ee906dbc6a5f3f9f5cf06e49edb005cc2d6d2c75

echo "Running old_run.py..."
python old_run.py
