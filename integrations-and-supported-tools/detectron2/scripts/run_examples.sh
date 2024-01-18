set -e

echo "Installing requirements..."
pip install -U -r requirements.txt
pip install -U 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

echo "Downloading and unzipping the dataset"
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip balloon_dataset.zip > /dev/null

echo "Running Neptune_detectron2.py..."
python Neptune_detectron2.py
