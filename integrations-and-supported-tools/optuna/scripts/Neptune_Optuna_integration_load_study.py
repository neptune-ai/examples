import lightgbm as lgb
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def objective(trial):
    data, target = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        'verbose': -1,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.2, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.2, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(test_x)
    accuracy = roc_auc_score(test_y, preds)

    return accuracy


# Log a Study to Neptune Run
new_run = neptune.init(api_token='ANONYMOUS', project='common/optuna-integration')  # you can pass your credentials here
neptune_callback = optuna_utils.NeptuneCallback(new_run)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, callbacks=[neptune_callback])

# Get the Run ID
run_id = new_run['sys/id'].fetch()

# stop the Run
new_run.stop()

# Fetch an existing Neptune Run where you logged the Optuna Study
existing_run = neptune.init(api_token='ANONYMOUS',
                            project='common/optuna-integration',
                            run=run_id)  # you can pass your credentials and Run ID here

# Load the Optuna Study from Neptune Run
study = optuna_utils.load_study_from_run(existing_run)

# Continue logging to the existing Neptne Run
neptune_callback = optuna_utils.NeptuneCallback(existing_run)
study.optimize(objective, n_trials=10, callbacks=[neptune_callback])

# Stop logging to a Neptune Run
existing_run.stop()
