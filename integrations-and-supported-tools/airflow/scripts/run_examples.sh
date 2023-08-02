# set -e

echo "Installing requirements..."
pip install -U -r requirements.txt --no-cache-dir

echo "Running Neptune_Airflow.py..."
python Neptune_Airflow.py
