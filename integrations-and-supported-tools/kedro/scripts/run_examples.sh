echo "Installing requirements..."
pip install -r requirements.txt

echo "Running kedro_neptune_quickstart..."
cd kedro_neptune_quickstart
#kedro run

echo "Running kedro_neptune_advanced..."
cd ../kedro_neptune_advanced
kedro run

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "