set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running reproduce_run.py..."
python reproduce_run.py

# You can get the commit ID to checkout to in source_code/git namespace of the run
# echo "Checkout to commit ID 146841173bb1f05b00c587f689695d041fdc55c3 ..."
# git checkout 146841173bb1f05b00c587f689695d041fdc55c3

echo "Running old_run.py..."
python old_run.py
