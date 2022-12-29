set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Keras.py..."
python Neptune_Keras.py

echo "Running Neptune_Keras_more_options.py..."
python Neptune_Keras_more_options.py
