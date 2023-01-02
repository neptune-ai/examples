set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_LightGBM_train.py..."
python Neptune_LightGBM_train.py

echo "Running Neptune_LightGBM_train_summary.py..."
python Neptune_LightGBM_train_summary.py

echo "Running Neptune_LightGBM_cv.py..."
python Neptune_LightGBM_cv.py

echo "Running Neptune_LightGBM_sklearn_api.py..."
python Neptune_LightGBM_sklearn_api.py
