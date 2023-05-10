set -e

echo "Installing requirements..."
pip install -U --user -r requirements.txt

echo "Running Neptune_Evidently_reports.py..."
python Neptune_Evidently_reports.py

echo "Running Neptune_Evidently_drifts.py..."
curl https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip --create-dirs -o data/Bike-Sharing-Dataset.zip
unzip -o data/Bike-Sharing-Dataset.zip -d data
python Neptune_Evidently_drifts.py
