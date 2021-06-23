echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Neptune_Catalyst.py..."
python Neptune_Catalyst.py

echo "Running Neptune_Catalyst_more_options.py..."
python Neptune_Catalyst_more_options.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "