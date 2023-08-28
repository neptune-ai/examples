set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Setting Neptune credentials..."
export NEPTUNE_PROJECT=common/huggingface-integration

echo "Running Neptune_Transformers.py..."
python Neptune_Transformers.py

echo "Running Neptune_Transformers_report_to.py..."
python Neptune_Transformers_report_to.py
