set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_XGBoost_train.py..."
python Neptune_XGBoost_train.py

echo "Running Neptune_XGBoost_cv.py..."
python Neptune_XGBoost_cv.py

echo "Running Neptune_XGBoost_sklearn_api.py..."
python Neptune_XGBoost_sklearn_api.py
