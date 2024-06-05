set -e

echo "Installing requirements..."
pip install -q -U -r requirements.txt

echo "Running hello_neptune.py..."
curl -o sample.png https://neptune.ai/wp-content/uploads/2024/05/blog_feature_image_046799_8_3_7_3-4.jpg
python hello_neptune.py
