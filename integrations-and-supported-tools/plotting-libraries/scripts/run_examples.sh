set -e

echo "Installing requirements..."
pip install -U -q -r requirements.txt

echo "Running Neptune_Plotting_Support.py..."
python Neptune_Plotting_Support.py
