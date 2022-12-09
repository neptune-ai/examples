echo "Installing requirements..."
pip install -r requirements.txt

echo "Downloading data..."
wget 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

echo "Running Neptune_TensorFlow_Keras.py..."
python Neptune_TensorFlow.py
