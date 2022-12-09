echo "Installing requirements..."
pip install -r requirements.txt

echo "Downloading data..."
wget 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

echo "Running Neptune_Tensorflow.py..."
python Neptune_Tensorflow.py
