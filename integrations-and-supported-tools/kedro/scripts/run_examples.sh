set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running kedro-neptune-quickstart..."
cd kedro-neptune-quickstart
kedro neptune init --api-token $NEPTUNE_API_TOKEN --project common/kedro-integration
kedro run

echo "Running kedro-neptune-advanced..."
cd ../kedro-neptune-advanced
kedro neptune init --api-token $NEPTUNE_API_TOKEN --project common/kedro-integration
kedro run
