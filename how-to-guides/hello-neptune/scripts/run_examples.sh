set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running hello_neptune.py..."
curl -O https://neptune.ai/wp-content/uploads/2023/06/Lenna_test_image.png
python hello_neptune.py
