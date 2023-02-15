set -e

echo "Installing requirements..."
pip install -U -r ../requirements.txt

echo "Running run_ml_baseline.py..."
python run_ml_baseline.py

echo "Running run_ml_prophet.py..."
python run_ml_prophet.py

echo "Running run_dl_lstm.py..."
python run_dl_lstm.py

echo "Running run_dl_lstm_finetune.py..."
python run_dl_lstm_finetune.py
