echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Neptune_Scikit_learn_regression.py..."
python Neptune_Scikit_learn_regression.py

echo "Running Neptune_Scikit_learn_classification.py..."
python Neptune_Scikit_learn_classification.py

echo "Running Neptune_Scikit_learn_clustering.py..."
python Neptune_Scikit_learn_clustering.py

echo "Running Neptune_Scikit_learn_other_options.py..."
python Neptune_Scikit_learn_other_options.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "