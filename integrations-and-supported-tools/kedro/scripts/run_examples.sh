set -e

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Setting Neptune credentials..."
export NEPTUNE_PROJECT=common/kedro-integration

echo "Running kedro-neptune-quickstart..."
cd kedro-neptune-quickstart
kedro neptune init --api-token $NEPTUNE_API_TOKEN --project $NEPTUNE_PROJECT
kedro run

echo "Running kedro-neptune-advanced..."
cd ../kedro-neptune-advanced
kedro neptune init --api-token $NEPTUNE_API_TOKEN --project $NEPTUNE_PROJECT
kedro run
