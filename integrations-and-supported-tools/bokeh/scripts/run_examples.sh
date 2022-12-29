set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Neptune_Bokeh_Support.py..."
python Neptune_Bokeh_Support.py
