set -e

echo "Installing requirements..."
pip install -r requirements.txt

echo "Setting Neptune credentials..."
export NEPTUNE_API_TOKEN=ANONYMOUS
export NEPTUNE_PROJECT=common/kedro-integration

echo "Running kedro_neptune_quickstart..."
cd kedro_neptune_quickstart
kedro neptune init --api-token $NEPTUNE_API_TOKEN --project $NEPTUNE_PROJECT
kedro run

echo "Running kedro_neptune_advanced..."
cd ../kedro_neptune_advanced
kedro neptune init --api-token $NEPTUNE_API_TOKEN --project $NEPTUNE_PROJECT
kedro run
