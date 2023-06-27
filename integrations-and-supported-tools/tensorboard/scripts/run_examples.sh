set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Tensorflow_Tensorboard.py..."
python Neptune_Tensorflow_Tensorboard.py

echo "Exporting previous logs..."
neptune tensorboard logs --project common/tensorboard-integration
