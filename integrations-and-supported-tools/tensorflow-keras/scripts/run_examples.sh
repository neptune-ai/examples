echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Neptune_TensorFlow_Keras.py..."
python Neptune_TensorFlow_Keras.py

echo "Running Neptune_TensorFlow_Keras_more_options.py..."
python Neptune_TensorFlow_Keras_more_options.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "