echo "Installing requirements..."
pip install -r requirements.txt

echo "Running re_run_failed_training.py..."
python re_run_failed_training.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "