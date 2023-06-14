set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running kedro-neptune-quickstart..."
cd kedro-neptune-quickstart
kedro neptune init --project common/kedro-integration --api-token "ANONYMOUS" # Replace with your own token or use $NEPTUNE_API_TOKEN
kedro run

echo "Running kedro-neptune-advanced..."
cd ../kedro-neptune-advanced
kedro neptune init --project common/kedro-integration --api-token "ANONYMOUS" # Replace with your own token or use $NEPTUNE_API_TOKEN
kedro run
