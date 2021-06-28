echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Neptune_Sacred.py..."
python Neptune_Sacred.py

echo "Running Neptune_Sacred_more_options.py..."
python Neptune_Sacred_more_options.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "