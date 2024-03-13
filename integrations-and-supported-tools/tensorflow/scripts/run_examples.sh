set -e

echo "Installing requirements..."
pip install -U -q -r requirements.txt

echo "Running Neptune_Tensorflow.py..."
python Neptune_Tensorflow.py
