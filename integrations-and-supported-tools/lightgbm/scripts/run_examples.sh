set -e
pip install -r requirements.txt
python Neptune_LightGBM_train.py
python Neptune_LightGBM_train_summary.py
python Neptune_LightGBM_cv.py
python Neptune_LightGBM_sklearn_api.py
