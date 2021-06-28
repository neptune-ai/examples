echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Organize_ML_runs.py..."
python Organize_ML_runs.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "