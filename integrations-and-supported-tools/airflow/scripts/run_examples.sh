set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Initialize Airflow DB..."
airflow initdb

echo "Running Neptune_Airflow.py..."
python Neptune_Airflow.py
