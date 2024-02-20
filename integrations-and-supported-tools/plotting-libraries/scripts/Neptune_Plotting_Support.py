# Import neptune and initialize a new run
import neptune

run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/plotting",
    tags=["script"],
)

# Log Altair charts to Neptune

## Create a sample chart
import altair as alt
from vega_datasets import data

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

## Log interactive chart to Neptune
run["altair"].upload(chart)

# Log Bokeh charts to Neptune
## Create a sample chart
import numpy as np
from bokeh.plotting import figure, output_notebook, show

N = 500
x = np.linspace(0, 10, N)
y = np.linspace(0, 10, N)
xx, yy = np.meshgrid(x, y)
d = np.sin(xx) * np.cos(yy)

p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
p.x_range.range_padding = p.y_range.range_padding = 0

p.image(image=[d], x=0, y=0, dw=10, dh=10, palette="Spectral11", level="image")
p.grid.grid_line_width = 0.5

## Log interactive chart to Neptune
run["bokeh"].upload(p)

# Log folium (leaflet) maps to Neptune
## Create a sample map
import folium

m = folium.Map()

## Log interacive map to Neptune
m.save("map.html")

run["folium"].upload("map.html")

# Log matplotlib charts to Neptune
## Create a sample chart
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.random.randn(2, 100)

figure, ax = plt.subplots(2, 2, figsize=(5, 5))
ax[0, 0].hist(data[0])
ax[1, 0].scatter(data[0], data[1])
ax[0, 1].plot(data[0], data[1])

## Log static chart to Neptune
run["matplotlib-static"].upload(figure)

## Log interactive chart to Neptune
from neptune.types import File

run["matplotlib-interactive"].upload(File.as_html(figure))

# Log Plotly charts to Neptune
## Create a sample chart
import plotly.express as px

df = px.data.iris()
fig = px.scatter_3d(df, x="sepal_length", y="sepal_width", z="petal_width", color="species")

## Log interactive chart to Neptune
run["plotly"].upload(fig)

# Log Seaborn charts to Neptune
## Create a sample chart
import seaborn as sns

df = sns.load_dataset("penguins")
plot = sns.pairplot(df, hue="species")

## Log chart to Neptune
run["seaborn"].upload(plot)

# Stop Neptune run
run.stop()

# Explore the charts in Neptune
# The charts can be found in the **All metadata** section.
# You can also explore this example run: https://app.neptune.ai/o/showcase/org/plotting/runs/details?viewId=standard-view&detailsTab=metadata&shortId=PLOT-2&path=&attribute=altair
