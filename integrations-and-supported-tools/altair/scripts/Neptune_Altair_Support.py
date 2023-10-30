# Import necessary libraries
import altair as alt
import neptune
from vega_datasets import data

# Initialize Neptune and create a new run
run = neptune.init_run(api_token=neptune.ANONYMOUS_API_TOKEN, project="common/altair-support")

# Create a sample chart
source = data.cars()

brush = alt.selection_interval()

points = (
    alt.Chart(source)
    .mark_point()
    .encode(
        x="Horsepower:Q",
        y="Miles_per_Gallon:Q",
        color=alt.condition(brush, "Origin:N", alt.value("lightgray")),
    )
    .add_params(brush)
)

bars = (
    alt.Chart(source)
    .mark_bar()
    .encode(y="Origin:N", color="Origin:N", x="count(Origin):Q")
    .transform_filter(brush)
)

chart = points & bars

# Log interactive image to Neptune
run["interactive_img"].upload(chart)

# Tracking will stop automatically once script execution is complete
