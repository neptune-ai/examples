set -e
pip install -r requirements.txt
python Neptune_XGBoost_train.py
python Neptune_XGBoost_cv.py
python Neptune_XGBoost_sklearn_api.py
