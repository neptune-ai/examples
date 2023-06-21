set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Keras_Tensorboard.py..."
python Neptune_Keras_Tensorboard.py

echo "Running Neptune_Pytorch_Tensorboard.py..."
python Neptune_Pytorch_Tensorboard.py

echo "Running Neptune_Tensorboardx.py..."
python Neptune_Tensorboardx.py

echo "Running Neptune_Tensorflow_Tensorboard.py..."
python Neptune_Tensorflow_Tensorboard.py
