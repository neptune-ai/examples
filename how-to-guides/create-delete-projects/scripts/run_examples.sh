set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running create_delete_projects.py..."
python create_delete_projects.py
