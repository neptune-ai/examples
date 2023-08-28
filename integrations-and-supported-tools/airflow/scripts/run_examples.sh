set -e

echo "Installing Airflow with constraints..."
pip install "apache-airflow==2.6.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.6.3/constraints-3.7.txt"

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Initialize Airflow DB..."
airflow db init

echo "Running Neptune_Airflow.py..."
python Neptune_Airflow.py
