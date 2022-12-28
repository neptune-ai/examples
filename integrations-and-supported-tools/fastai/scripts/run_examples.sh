set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_fastai.py..."
python Neptune_fastai.py

echo "Running Neptune_fastai_more_options.py..."
python Neptune_fastai_more_options.py
