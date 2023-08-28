set -e

export NEPTUNE_CUSTOM_RUN_ID=`date +"%Y%m%d%H%M%s%N"`
export NEPTUNE_PROJECT="common/pipelining-support"

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running data_preprocessing.py..."
python data_preprocessing.py

echo "Running model_training.py..."
python model_training.py

echo "Running model_validation.py..."
python model_validation.py

echo "Running model_promotion.py..."
python model_promotion.py
