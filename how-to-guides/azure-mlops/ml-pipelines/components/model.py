from typing import List

import xgboost as xgb


def load_xgboost_model(callbacks: List = None, checkpoint=None, random_state: int = 42):
    if checkpoint:
        model = xgb.XGBRegressor(random_state=random_state)
        model.load_model(checkpoint)
    else:
        model = xgb.XGBRegressor(random_state=random_state, callbacks=callbacks)
    return model
