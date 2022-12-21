set -e

echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Neptune_Plotly_Support.py..."
python Neptune_Plotly_Support.py
