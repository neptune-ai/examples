set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running Kedro spaceflights..."
cd spaceflights-pandas
kedro neptune init --project common/kedro-integration --api-token "ANONYMOUS" # Replace with your own token and project
kedro run
