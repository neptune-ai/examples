echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Neptune_Optuna_integration_quickstart.py..."
python Neptune_Optuna_integration_quickstart.py

echo "Running Neptune_Optuna_integration_customize_callback.py..."
python Neptune_Optuna_integration_customize_callback.py

echo "Running Neptune_Optuna_integration_log_after_study.py..."
python Neptune_Optuna_integration_log_after_study.py

echo "Running Neptune_Optuna_integration_log_study_and_trial_level.py..."
python Neptune_Optuna_integration_log_study_and_trial_level.py

echo "Running Neptune_Optuna_integration_load_study.py..."
python Neptune_Optuna_integration_load_study.py

# Uncomment the next line to prevent execution window from closing when script execution is complete
# read -p "Press [Enter] to exit "