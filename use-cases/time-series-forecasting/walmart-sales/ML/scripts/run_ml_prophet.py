import matplotlib.pyplot as plt
import neptune.new as neptune
import neptune.new.integrations.prophet as npt_utils
import seaborn as sns
from neptune.new.types import File
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import *

sns.set()
plt.rcParams["figure.figsize"] = 15, 8
plt.rcParams["image.cmap"] = "viridis"
plt.ioff()


def main():
    # (neptune) Initialize Neptune run
    run = neptune.init_run(tags=["prophet", "walmart-sales"])

    DATA_PATH = "./sales/data"

    # Load dataset
    df = load_data(DATA_PATH, cache=True)

    # Normalize sales data
    df_normalized = normalize_data(df, "Weekly_Sales")

    # Encode categorical data
    df_encoded = df_normalized.copy()
    df_encoded = encode_categorical_data(df_encoded)

    # Create Lagged features
    df_encoded = create_lags(df_encoded)

    # Get train data
    X_train, X_valid, y_train, y_valid = get_train_data(
        df_encoded[df_encoded.Dept == 1], ["Weekly_Sales", "Year"]
    )
    prophet_data = get_prophet_data_format(X_train, y_train)

    # Train model
    model = Prophet()
    model.add_country_holidays(country_name="US")
    model.fit(prophet_data)

    run["model_config"] = npt_utils.get_model_config(model)

    future_prophet_data = get_prophet_data_format(X_valid, y_valid)
    forecast = model.predict(future_prophet_data)
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()

    # Calculate scores
    r2 = r2_score(y_valid, forecast.yhat)
    rmse = mean_squared_error(y_valid, forecast.yhat, squared=False)
    mae = mean_absolute_error(y_valid, forecast.yhat)

    # (neptune) Log scores
    run["training/val/r2"] = r2
    run["training/val/rmse"] = rmse
    run["training/val/mae"] = mae

    # Create predicitions visualizations
    fig1 = model.plot(forecast)

    # (neptune) Log predictions visualizations
    run["forecast_plots"].upload(File.as_image(fig1))
    run["forecast_components"] = npt_utils.get_forecast_components(model, forecast)


if __name__ == "__main__":
    main()
