set -e

echo "Installing requirements..."
pip install --q -U -r requirements.txt

echo "Initializing ZenML..."
zenml init

echo "Installing ZenML's Neptune integration..."
zenml integration install neptune -y

echo "Registering Neptune as ZenML experiment tracker..."
zenml experiment-tracker register neptune_tracker \
    --flavor=neptune \
    --project="common/zenml" # Replace with your own project

echo "Creating new ZenML stack with Neptune tracking..."
zenml stack register neptune_stack -a default -o default -e neptune_tracker --set

echo "Running Neptune_ZenML.py..."
python Neptune_ZenML.py
