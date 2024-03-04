set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

if [[ "$OSTYPE" == "darwin"* ]]; then
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
else
pip install -U 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
fi

echo "Downloading and unzipping the dataset"
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip balloon_dataset.zip > /dev/null

echo "Running Neptune_detectron2.py..."
python Neptune_detectron2.py
