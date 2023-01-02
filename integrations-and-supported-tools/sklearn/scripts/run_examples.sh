set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Scikit_learn_regression.py..."
python Neptune_Scikit_learn_regression.py

echo "Running Neptune_Scikit_learn_classification.py..."
python Neptune_Scikit_learn_classification.py

echo "Running Neptune_Scikit_learn_clustering.py..."
python Neptune_Scikit_learn_clustering.py

echo "Running Neptune_Scikit_learn_other_options.py..."
python Neptune_Scikit_learn_other_options.py
