set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Tensorflow.py..."
python Neptune_Tensorflow.py
