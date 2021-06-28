echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Neptune_XGBoost_train.py..."
python Neptune_XGBoost_train.py

echo "Running Neptune_XGBoost_cv.py..."
python Neptune_XGBoost_cv.py

echo "Running Neptune_XGBoost_sklearn_api.py..."
python Neptune_XGBoost_sklearn_api.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "