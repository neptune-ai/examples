set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Tensorflow_Tensorboard.py..."
python Neptune_Tensorflow_Tensorboard.py
