import matplotlib.pyplot as plt
import neptune.new as neptune
import seaborn as sns
from neptune.new.types import File
from utils import *

sns.set()
plt.rcParams["figure.figsize"] = 15, 8
plt.rcParams["image.cmap"] = "viridis"
plt.ioff()


def main():

    # (neptune) Initialize Neptune run
    run = neptune.init_run(tags=["eda", "walmart-sales", "showcase-run"])

    DATA_PATH = "./sales/data"

    # Load dataset
    df = load_data(DATA_PATH, cache=True)

    # Create data statistics
    df_statistics = df.describe()

    # Create data processing
    total_no_days = (
        pd.to_datetime(df.Date.max()).date() - pd.to_datetime(df.Date.min()).date()
    ).days
    total_no_months = total_no_days * 0.032855
    total_no_years = total_no_months * 0.0833334
    abs_no_years = int(total_no_years)
    sales_warm_vs_cold = (
        df[df.Temperature > 15]["Weekly_Sales"].sum()
        - df[df.Temperature < 15]["Weekly_Sales"].sum()
    )

    # (neptune) Log dataset statistics & processing
    run["data/statistics/dataset_statistics"].upload(File.as_html(df_statistics))
    run["data/statistics/total_no_days"] = total_no_days
    run["data/statistics/total_no_years"] = total_no_years
    run["data/statistics/abs_no_years"] = abs_no_years
    run["data/statistics/diffSales_warm_vs_cold"] = "$ {:,}".format(round(sales_warm_vs_cold))

    # Create data visualizations
    corr_matrix = df.corr()
    plt.figure()
    sns_corr_matrix = sns.heatmap(data=corr_matrix, annot=True)

    plt.figure()
    sales_vs_week = sns.lineplot(
        x="Week", y="Weekly_Sales", hue="Year", data=df, palette=["b", "g", "orange"]
    )

    plt.figure()
    sales_vs_month = sns.lineplot(
        x="Month", y="Weekly_Sales", hue="Year", data=df, palette=["b", "g", "orange"]
    )

    plt.figure()
    sales_vs_qt = sns.lineplot(
        x=df["Date"].dt.quarter,
        y="Weekly_Sales",
        hue="Year",
        data=df,
        palette=["b", "g", "orange"],
    )

    plt.figure()
    sales_vs_isholiday = sns.lineplot(x="Month", y="Weekly_Sales", hue="IsHoliday", data=df)

    plt.figure()
    sales_vs_store = sns.lineplot(
        x="Store", y="Weekly_Sales", hue="Year", data=df, palette=["b", "g", "orange"]
    )

    plt.figure()
    sales_vs_dept = sns.lineplot(
        x="Dept", y="Weekly_Sales", hue="Year", data=df, palette=["b", "g", "orange"]
    )

    plt.figure()
    sales_vs_storeType = sns.boxenplot(
        x="Type", y="Weekly_Sales", data=df, hue="IsHoliday", palette=["b", "orange"]
    )

    plt.figure()
    sales_vs_temp = sns.histplot(data=df.groupby("Weekly_Sales")["Temperature"].mean(), bins=30)

    df_normalized = normalize_data(df, "Weekly_Sales")
    plt.figure()
    sales_vs_storeType_normalized = sns.boxenplot(
        x="Type",
        y="Weekly_Sales",
        data=df_normalized,
        hue="IsHoliday",
        palette=["b", "orange"],
    )

    # (neptune) Log data visualizations
    run["data/visualizations/corr_matrix"].upload(File.as_image(sns_corr_matrix.figure))
    run["data/visualizations/sales_vs_week"].upload(File.as_image(sales_vs_week.figure))
    run["data/visualizations/sales_vs_month"].upload(File.as_image(sales_vs_month.figure))
    run["data/visualizations/sales_vs_quarter"].upload(File.as_image(sales_vs_qt.figure))
    run["data/visualizations/sales_vs_store"].upload(File.as_image(sales_vs_store.figure))
    run["data/visualizations/sales_vs_dept"].upload(File.as_image(sales_vs_dept.figure))
    run["data/visualizations/sales_vs_IsHoliday"].upload(File.as_image(sales_vs_isholiday.figure))
    run["data/visualizations/sales_vs_temperature"].upload(File.as_image(sales_vs_temp.figure))
    run["data/visualizations/sales_vs_storeType"].upload(File.as_image(sales_vs_storeType.figure))
    run["data/visualizations/sales_vs_storeType_normalized"].upload(
        File.as_image(sales_vs_storeType_normalized.figure)
    )


if __name__ == "__main__":
    main()
