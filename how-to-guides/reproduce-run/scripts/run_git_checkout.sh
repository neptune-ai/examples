set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Checkout to commit ID b6ea01d27dd4c10fec12f4d49220ee34cc03fb3c ..."
git checkout b6ea01d27dd4c10fec12f4d49220ee34cc03fb3c

echo "Running old_run.py..."
python old_run.py
