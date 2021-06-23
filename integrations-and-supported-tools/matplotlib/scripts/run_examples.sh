echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Neptune_Matplotlib_Support.py..."
python Neptune_Matplotlib_Support.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "