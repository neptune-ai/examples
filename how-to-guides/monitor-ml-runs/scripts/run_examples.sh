echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Monitor_ML_runs_live.py..."
python Monitor_ML_runs_live.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "