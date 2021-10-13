echo "Installing requirements..."
pip install -r requirements.txt

echo "Running create_delete_projects.py..."
python create_delete_projects.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "