set -e

echo "Installing requirements..."
pip install -U -q -r requirements.txt

echo "Creating sample MLflow runs for export..."

echo "Running train_keras_mlflow.py..."
python train_keras_mlflow.py

echo "Running train_scikit_mlflow.py..."
python train_scikit_mlflow.py

echo "Exporting MLflow runs to Neptune..."
neptune mlflow --project common/mlflow-integration

echo "Tracking MLflow run in Neptune using Neptune tracking URI..."
python mlflow_neptune_plugin.py
