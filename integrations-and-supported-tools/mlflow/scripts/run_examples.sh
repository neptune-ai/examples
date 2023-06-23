set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running train_keras_mlflow.py..."
python train_keras_mlflow.py

echo "Running train_scikit_mlflow.py..."
python train_scikit_mlflow.py

echo "Export MLFlow runs to Neptune..."
neptune mlflow --project common/quickstarts
